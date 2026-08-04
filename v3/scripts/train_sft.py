import os
# CUDA 초기화 전에 메모리 파편화 완화 설정 적용
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
import re
import sys
import argparse
import itertools
from tqdm import tqdm

# scripts 폴더 임포트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet, KoJamoTransformer
from dataset import KoJamoDataset
from config import EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
from chat import infer_checkpoint_spec

# ── SFT 하이퍼파라미터 ─────────────────────────────────────
SFT_LR = 0.00005  # SFT에서 pretrain 지식 망각을 줄이는 낮은 학습률
SFT_EPOCHS = 5
BATCH_SIZE = 8    # 길이 기반 버켓팅을 통해 VRAM 오버헤드를 낮췄으므로 안전하게 8로 복구
# ─────────────────────────────────────────────────────────


def apply_sft_input_dropout(x, mask_y, probability=0.05):
    """Mask answer-side inputs only to reduce teacher-forcing exposure bias."""
    if probability <= 0:
        return x
    answer_positions = mask_y > 0
    drop = (torch.rand_like(mask_y) < probability) & answer_positions
    return x.masked_fill(drop.unsqueeze(-1), 0)


def build_sft_model(vocab_sizes, checkpoint_path=None, device=None):
    """Create the SFT model and optionally load any compatible pretrain checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if state_dict is None:
        model = KoJamoNet(
            vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        )
    else:
        spec = infer_checkpoint_spec(state_dict)
        if spec["variant"] == "transformer":
            model = KoJamoTransformer(
                vocab_sizes=vocab_sizes, emb_dim=spec["emb_dim"],
                hidden_dim=spec["hidden_dim"], num_layers=spec["num_layers"],
                num_heads=spec["num_heads"], dropout=DROPOUT,
                max_seq_length=spec["max_seq_length"],
            )
        else:
            model = KoJamoNet(
                vocab_sizes=vocab_sizes, emb_dim=spec["emb_dim"],
                hidden_dim=spec["hidden_dim"], num_layers=spec["num_layers"],
                dropout=DROPOUT, cascade=spec["variant"] == "cascade",
            )
        model.load_state_dict(state_dict)

    return model.to(device)

class LengthGroupedSampler(torch.utils.data.Sampler):
    """지문 길이를 기준으로 샘플을 정렬하여 배치 내 패딩(PAD) 낭비를 최소화함 (버켓팅)."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # 각 샘플의 길이를 측정하여 로컬 인덱스(0 ~ len-1) 정렬 (Subset 대응)
        if hasattr(dataset, 'dataset'):
            self.indices = sorted(range(len(dataset)), key=lambda local_idx: len(dataset.dataset.samples[dataset.indices[local_idx]]))
        else:
            self.indices = sorted(range(len(dataset)), key=lambda idx: len(dataset.samples[idx]))
        
    def __iter__(self):
        # 정렬된 인덱스를 배치 단위로 쪼갬
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        # 에폭 시작 시 배치 순서만 셔플하여 다양한 데이터를 골고루 학습하게 함
        import random
        random.shuffle(batches)
        
        # 셔플된 배치를 단일 차원의 인덱스 리스트로 평탄화
        flat_indices = []
        for batch in batches:
            flat_indices.extend(batch)
        return iter(flat_indices)
        
    def __len__(self):
        return len(self.dataset)


def save_sft_checkpoint(model, prefix="model_sft_v"):
    os.makedirs("checkpoints", exist_ok=True)
    existing = glob.glob(f"checkpoints/{prefix}*.pth")
    pattern = re.escape(prefix) + r"(\d+)\.pth"
    nums = [int(re.search(pattern, f).group(1)) for f in existing if re.search(pattern, f)]
    save_path = f"checkpoints/{prefix}{max(nums) + 1 if nums else 1}.pth"
    # torch.compile 대응: 원본 모듈의 state_dict 저장
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), save_path)
    print(f"💾 SFT 가중치 저장 완료: {save_path}")
    return save_path


def sft_collate_fn(batch, max_seq_length=None):
    # batch: list of tuples (x, y, mask_y)
    if max_seq_length is not None:
        batch = [
            (x[:max_seq_length], y[:max_seq_length], mask_y[:max_seq_length])
            for x, y, mask_y in batch
        ]
    lengths = [x.size(0) for x, y, mask_y in batch]
    max_len = max(lengths)
    
    padded_xs = []
    padded_ys = []
    padded_masks = []
    for x, y, mask_y in batch:
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            # 0은 모든 트랙의 PAD 토큰값
            pad_x = torch.zeros(pad_len, 6, dtype=x.dtype)
            pad_y = torch.zeros(pad_len, 6, dtype=y.dtype)
            pad_mask = torch.zeros(pad_len, dtype=mask_y.dtype) # 패딩 토큰은 loss_mask를 0.0으로 설정
            padded_xs.append(torch.cat([x, pad_x], dim=0))
            padded_ys.append(torch.cat([y, pad_y], dim=0))
            padded_masks.append(torch.cat([mask_y, pad_mask], dim=0))
        else:
            padded_xs.append(x)
            padded_ys.append(y)
            padded_masks.append(mask_y)
            
    return torch.stack(padded_xs, dim=0), torch.stack(padded_ys, dim=0), torch.stack(padded_masks, dim=0)


def train_sft(checkpoint_path=None, epochs=SFT_EPOCHS, batch_size=BATCH_SIZE,
              limit_batches=None, data_dir="train_data_sft",
              checkpoint_prefix="transformer_sft_v", patience=2,
              input_dropout=0.05, learning_rate=SFT_LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(" 🛠️ Ko-JamoNet v2 — Supervised Fine-Tuning (SFT) 개시")
    print("=" * 60)
    print(f"🚀 디바이스: {device}")
    print(f"📁 학습 대상: {data_dir}/ 폴더 (Q:/A: 턴 구조 보존된 대화 데이터)")
    print(f"⏳ 총 학습 에폭: {epochs}")

    # train_data(프리트레인용 프로즈)와 분리된 별도 폴더 — Q:/A: 턴 구조가 실제로 보존된 데이터라야
    # "형식 맞추기"가 의미 있음. SFT 모드로 개별 지문별로 나누어 로드 및 동적 패딩 적용.
    full_dataset = KoJamoDataset(data_dir=data_dir, seq_length=1000, stride=100, is_sft=True)
    
    # 9:1 비율로 Train / Validation 분할 (과적합 실시간 감지용)
    train_size = int(0.9 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_sampler  = LengthGroupedSampler(train_dataset, batch_size=batch_size)
    val_sampler    = LengthGroupedSampler(val_dataset, batch_size=batch_size)
    
    collate_fn = lambda batch: sft_collate_fn(batch, max_seq_length=512)
    dataloader     = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    vocab_sizes = full_dataset.tokenizer.get_vocab_sizes()
 
    # SFT 가중치가 이미 존재하면 이어서 학습 (우선순위 1)
    sft_files = glob.glob(f"checkpoints/{checkpoint_prefix}*.pth") if checkpoint_path is None else []
    prefix_pattern = re.escape(checkpoint_prefix) + r"(\d+)\.pth"
    sft_nums  = [(int(re.search(prefix_pattern, f).group(1)), f)
                 for f in sft_files if re.search(prefix_pattern, f)]
                 
    if sft_nums:
        _, latest_sft = max(sft_nums, key=lambda x: x[0])
        print(f"🔄 이전 SFT 가중치 이어서 학습 (우선순위 1): {latest_sft}")
        model = build_sft_model(vocab_sizes, latest_sft, device)
    else:
        # SFT 가중치가 없으면 지정 checkpoint 또는 best Transformer를 우선 사용
        if checkpoint_path is None and os.path.exists("checkpoints/unattended_full_best.pth"):
            checkpoint_path = "checkpoints/unattended_full_best.pth"
        if checkpoint_path is None:
            pretrain_files = glob.glob("checkpoints/model_v*.pth")
            pretrain_nums  = [(int(re.search(r"model_v(\d+)\.pth", f).group(1)), f)
                              for f in pretrain_files if re.search(r"model_v(\d+)\.pth", f)]
            checkpoint_path = max(pretrain_nums, key=lambda x: x[0])[1] if pretrain_nums else None
        if checkpoint_path:
            print(f"🔄 1단계 Pretrain 가중치 로드 완료: {checkpoint_path}")
        model = build_sft_model(vocab_sizes, checkpoint_path, device)
        if checkpoint_path is None:
            print("🚨 사전학습 및 SFT 파일이 발견되지 않아 처음부터 SFT 학습을 기동합니다.")

    # ⚡ PyTorch 2.x JIT 컴파일 적용: 커널 융합(Fusion) 및 Triton 자동 변환 가속
    # VRAM이 극도로 좁은 GPU(6GB) 환경에서 메모리 풀 오버헤드로 OOM이 재발할 수 있어 비활성화
    # if hasattr(torch, 'compile') and device.type == 'cuda':
    #     print("⚡ torch.compile을 사용하여 모델 연산 융합(Fusion) 가속 적용...")
    #     model = torch.compile(model)

    # SFT용 최적화 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # 100에폭에 걸쳐 LR을 안정적으로 감쇠시키는 CosineAnnealingLR 적용 (주기적 스파이크 충격 차단)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # 모델이 SFT 모드에서 log_final(로그 확률)을 반환하므로, CrossEntropy 대신 NLLLoss를 사용하는 것이 수학적으로 무결함
    criterion = nn.NLLLoss()
    criterion_none = nn.NLLLoss(reduction='none')

    # bfloat16 — float16+GradScaler 조합에서 pretrain 중 attention 가중치가 NaN으로 발산한
    # 사고가 있었음 (train.py와 동일한 문제). 여기도 동일하게 방지.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    try:
        for epoch in range(1, epochs + 1):
            total_loss = 0
            train_iter = dataloader if limit_batches is None else itertools.islice(dataloader, limit_batches)
            pbar = tqdm(train_iter, total=min(len(dataloader), limit_batches) if limit_batches else len(dataloader), desc=f"SFT Epoch {epoch}", leave=True)
            model.train()
            
            accumulation_steps = 4
            optimizer.zero_grad()
            optimizer_steps = 0
            
            for step, (x, y, mask_y) in enumerate(pbar):
                x, y, mask_y = x.to(device), y.to(device), mask_y.to(device)
                x_model = apply_sft_input_dropout(x, mask_y, input_dropout)
                
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=device.type == "cuda"):
                    logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num = model(x_model, target_for_forcing=y)
                    types_y = model._get_types(y)
                    
                    # --- 6트랙 타입 마스크 Loss 계산 ---
                    loss_type = criterion_none(logits_type.view(-1, 4), types_y.view(-1))
                    loss = (loss_type * mask_y.view(-1)).sum() / (mask_y.sum() + 1e-8)
                    
                    loss_cho = criterion_none(logits_cho.view(-1, logits_cho.size(-1)), y[:, :, 0].view(-1))
                    loss_jung = criterion_none(logits_jung.view(-1, logits_jung.size(-1)), y[:, :, 1].view(-1))
                    loss_jong = criterion_none(logits_jong.view(-1, logits_jong.size(-1)), y[:, :, 2].view(-1))
                    loss_sym = criterion_none(logits_sym.view(-1, logits_sym.size(-1)), y[:, :, 3].view(-1))
                    loss_eng = criterion_none(logits_eng.view(-1, logits_eng.size(-1)), y[:, :, 4].view(-1))
                    loss_num = criterion_none(logits_num.view(-1, logits_num.size(-1)), y[:, :, 5].view(-1))
                    
                    # 프롬프트 마스크(mask_y)를 곱하여 질문/지문 부분의 가중치 업데이트 차단
                    mask_korean = (types_y.view(-1) == 0).float() * mask_y.view(-1)
                    mask_sym = (types_y.view(-1) == 1).float() * mask_y.view(-1)
                    mask_eng = (types_y.view(-1) == 2).float() * mask_y.view(-1)
                    mask_num = (types_y.view(-1) == 3).float() * mask_y.view(-1)
                    
                    loss += (loss_cho * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                    loss += (loss_jung * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                    loss += (loss_jong * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                    loss += (loss_sym * mask_sym).sum() / (mask_sym.sum() + 1e-8)
                    loss += (loss_eng * mask_eng).sum() / (mask_eng.sum() + 1e-8)
                    loss += (loss_num * mask_num).sum() / (mask_num.sum() + 1e-8)
                    
                    # 누적 경사도 정규화
                    loss = loss / accumulation_steps
    
                if not torch.isfinite(loss):
                    print(f"⚠️ 비정상 loss 감지 — 이 스텝 건너뜀")
                    continue
    
                loss.backward()
                
                # accumulation_steps 마다 또는 데이터로더 마지막 스텝에서 가중치 업데이트
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1
    
                # 원래 척도 복구하여 로깅에 표기
                total_loss += loss.item() * accumulation_steps
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{(loss.item() * accumulation_steps):.4f}", lr=f"{current_lr:.6f}")
    
            avg = total_loss / max(1, pbar.n)
            
            # --- 실시간 과적합 검증 (Validation Loss 계산) ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_x, val_y, val_mask_y in val_dataloader:
                    val_x, val_y, val_mask_y = val_x.to(device), val_y.to(device), val_mask_y.to(device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=device.type == "cuda"):
                        v_logits_type, v_logits_cho, v_logits_jung, v_logits_jong, v_logits_sym, v_logits_eng, v_logits_num = model(val_x, target_for_forcing=val_y)
                        v_types_y = model._get_types(val_y)
                        
                        v_loss_type = criterion_none(v_logits_type.view(-1, 4), v_types_y.view(-1))
                        v_loss = (v_loss_type * val_mask_y.view(-1)).sum() / (val_mask_y.sum() + 1e-8)
                        
                        v_loss_cho = criterion_none(v_logits_cho.view(-1, v_logits_cho.size(-1)), val_y[:, :, 0].view(-1))
                        v_loss_jung = criterion_none(v_logits_jung.view(-1, v_logits_jung.size(-1)), val_y[:, :, 1].view(-1))
                        v_loss_jong = criterion_none(v_logits_jong.view(-1, v_logits_jong.size(-1)), val_y[:, :, 2].view(-1))
                        v_loss_sym = criterion_none(v_logits_sym.view(-1, v_logits_sym.size(-1)), val_y[:, :, 3].view(-1))
                        v_loss_eng = criterion_none(v_logits_eng.view(-1, v_logits_eng.size(-1)), val_y[:, :, 4].view(-1))
                        v_loss_num = criterion_none(v_logits_num.view(-1, v_logits_num.size(-1)), val_y[:, :, 5].view(-1))
                        
                        v_mask_korean = (v_types_y.view(-1) == 0).float() * val_mask_y.view(-1)
                        v_mask_sym = (v_types_y.view(-1) == 1).float() * val_mask_y.view(-1)
                        v_mask_eng = (v_types_y.view(-1) == 2).float() * val_mask_y.view(-1)
                        v_mask_num = (v_types_y.view(-1) == 3).float() * val_mask_y.view(-1)
                        
                        v_loss += (v_loss_cho * v_mask_korean).sum() / (v_mask_korean.sum() + 1e-8)
                        v_loss += (v_loss_jung * v_mask_korean).sum() / (v_mask_korean.sum() + 1e-8)
                        v_loss += (v_loss_jong * v_mask_korean).sum() / (v_mask_korean.sum() + 1e-8)
                        v_loss += (v_loss_sym * v_mask_sym).sum() / (v_mask_sym.sum() + 1e-8)
                        v_loss += (v_loss_eng * v_mask_eng).sum() / (v_mask_eng.sum() + 1e-8)
                        v_loss += (v_loss_num * v_mask_num).sum() / (v_mask_num.sum() + 1e-8)
                        
                        val_loss += v_loss.item()
                        
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"✨ SFT Epoch {epoch} 완료 | Train Loss: {avg:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # epoch checkpoint와 validation best checkpoint를 분리한다.
            save_sft_checkpoint(model, checkpoint_prefix)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                best_path = f"checkpoints/{checkpoint_prefix}best.pth"
                torch.save(raw_model.state_dict(), best_path)
                print(f"🏆 SFT best 저장: {best_path} (val={best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"🛑 validation {patience}회 연속 개선 없음 — 조기 종료")
                    break
            if optimizer_steps > 0:
                scheduler.step()
    except KeyboardInterrupt:
        print(f"\n🛑 SFT 학습 중단 요청 감지. {epoch}에폭 중간 상태 가중치를 저장합니다...")
        save_sft_checkpoint(model, checkpoint_prefix)
        print("👋 종료합니다.")

    print("🎉 SFT 모든 과정 완료! 최종 모델을 대화에 사용할 수 있습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ko-JamoNet Transformer/GRU SFT")
    parser.add_argument("--checkpoint", help="pretrain or SFT checkpoint to load")
    parser.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--data-dir", default="train_data_sft")
    parser.add_argument("--prefix", default="transformer_sft_v")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=SFT_LR)
    args = parser.parse_args()
    train_sft(
        args.checkpoint, args.epochs, args.batch, args.limit_batches,
        args.data_dir, args.prefix, args.patience,
        args.input_dropout, args.lr,
    )
