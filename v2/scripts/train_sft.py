import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import glob
import re
import sys
from tqdm import tqdm

# scripts 폴더 임포트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from dataset import KoJamoDataset
from config import EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

# ── SFT 하이퍼파라미터 ─────────────────────────────────────
SFT_LR = 0.0005   # Pretrain보다 낮은 학습률 설정 (기존 지식 유실 방지)
SFT_EPOCHS = 10   # SFT는 5~10 에폭이면 스타일 정렬에 충분함
BATCH_SIZE = 32
# ─────────────────────────────────────────────────────────

def save_sft_checkpoint(model):
    os.makedirs("checkpoints", exist_ok=True)
    existing = glob.glob("checkpoints/model_sft_v*.pth")
    nums     = [int(re.search(r"model_sft_v(\d+)\.pth", f).group(1))
                for f in existing if re.search(r"model_sft_v(\d+)\.pth", f)]
    save_path = f"checkpoints/model_sft_v{max(nums) + 1 if nums else 1}.pth"
    # torch.compile 대응: 원본 모듈의 state_dict 저장
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), save_path)
    print(f"💾 SFT 가중치 저장 완료: {save_path}")


def train_sft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(" 🛠️ Ko-JamoNet v2 — Supervised Fine-Tuning (SFT) 개시")
    print("=" * 60)
    print(f"🚀 디바이스: {device}")
    print(f"📁 학습 대상: train_data_sft/ 폴더 (Q:/A: 턴 구조 보존된 대화 데이터)")
    print(f"⏳ 총 학습 에폭: {SFT_EPOCHS}")

    # train_data(프리트레인용 프로즈)와 분리된 별도 폴더 — Q:/A: 턴 구조가 실제로 보존된 데이터라야
    # "형식 맞추기"가 의미 있음. stride=100으로 대화 흐름을 촘촘히 복습.
    dataset     = KoJamoDataset(data_dir="train_data_sft", seq_length=1000, stride=100)
    dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    vocab_sizes = dataset.tokenizer.get_vocab_sizes()

    model = KoJamoNet(vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
                      num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)

    # Pretrain 가중치 로드 (최신 model_v*.pth 검색)
    pretrain_files = glob.glob("checkpoints/model_v*.pth")
    pretrain_nums  = [(int(re.search(r"model_v(\d+)\.pth", f).group(1)), f)
                      for f in pretrain_files if re.search(r"model_v(\d+)\.pth", f)]
    
    if pretrain_nums:
        _, latest_pretrain = max(pretrain_nums, key=lambda x: x[0])
        print(f"🔄 1단계 Pretrain 가중치 로드 완료: {latest_pretrain}")
        checkpoint = torch.load(latest_pretrain, map_location=device)
        # 신규 포맷(model/optimizer/scheduler/epoch 딕셔너리)과 구 포맷(순수 state_dict) 모두 지원
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        # 기존 SFT 파일이 있을 경우 이어 학습 시도
        sft_files = glob.glob("checkpoints/model_sft_v*.pth")
        sft_nums  = [(int(re.search(r"model_sft_v(\d+)\.pth", f).group(1)), f)
                     for f in sft_files if re.search(r"model_sft_v(\d+)\.pth", f)]
        if sft_nums:
            _, latest_sft = max(sft_nums, key=lambda x: x[0])
            print(f"🔄 이전 SFT 가중치 이어서 학습: {latest_sft}")
            model.load_state_dict(torch.load(latest_sft, map_location=device))
        else:
            print("🚨 경고: 사전학습(Pretrain) 체크포인트를 찾을 수 없습니다.")
            print("   먼저 train.py를 실행하여 1단계 학습을 완료해 주세요.")
            return

    # ⚡ PyTorch 2.x JIT 컴파일 적용: 커널 융합(Fusion) 및 Triton 자동 변환 가속
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("⚡ torch.compile을 사용하여 모델 연산 융합(Fusion) 가속 적용...")
        model = torch.compile(model)

    # SFT용 최적화 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SFT_EPOCHS, eta_min=1e-5)
    
    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    # bfloat16 — float16+GradScaler 조합에서 pretrain 중 attention 가중치가 NaN으로 발산한
    # 사고가 있었음 (train.py와 동일한 문제). 여기도 동일하게 방지.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for epoch in range(1, SFT_EPOCHS + 1):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch}/{SFT_EPOCHS}", leave=True)
        model.train()
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=device.type == "cuda"):
                logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num = model(x, target_for_forcing=y)
                types_y = model._get_types(y)
                
                # --- 6트랙 타입 마스크 Loss 계산 (정적 형상 및 논블로킹 최적화) ---
                loss = criterion(logits_type.view(-1, 4), types_y.view(-1))
                
                loss_cho = criterion_none(logits_cho.view(-1, logits_cho.size(-1)), y[:, :, 0].view(-1))
                loss_jung = criterion_none(logits_jung.view(-1, logits_jung.size(-1)), y[:, :, 1].view(-1))
                loss_jong = criterion_none(logits_jong.view(-1, logits_jong.size(-1)), y[:, :, 2].view(-1))
                loss_sym = criterion_none(logits_sym.view(-1, logits_sym.size(-1)), y[:, :, 3].view(-1))
                loss_eng = criterion_none(logits_eng.view(-1, logits_eng.size(-1)), y[:, :, 4].view(-1))
                loss_num = criterion_none(logits_num.view(-1, logits_num.size(-1)), y[:, :, 5].view(-1))
                
                mask_korean = (types_y.view(-1) == 0).float()
                mask_sym = (types_y.view(-1) == 1).float()
                mask_eng = (types_y.view(-1) == 2).float()
                mask_num = (types_y.view(-1) == 3).float()
                
                loss += (loss_cho * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                loss += (loss_jung * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                loss += (loss_jong * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                loss += (loss_sym * mask_sym).sum() / (mask_sym.sum() + 1e-8)
                loss += (loss_eng * mask_eng).sum() / (mask_eng.sum() + 1e-8)
                loss += (loss_num * mask_num).sum() / (mask_num.sum() + 1e-8)

            if not torch.isfinite(loss):
                print(f"⚠️ 비정상 loss({loss.item()}) 감지 — 이 스텝 건너뜀")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        avg = total_loss / len(dataloader)
        print(f"✨ SFT Epoch {epoch} 완료 | Average Loss: {avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # SFT는 매 에폭마다 가중치 백업
        save_sft_checkpoint(model)
        scheduler.step()

    print("🎉 SFT 모든 과정 완료! 최종 모델을 대화에 사용할 수 있습니다.")

if __name__ == "__main__":
    train_sft()
