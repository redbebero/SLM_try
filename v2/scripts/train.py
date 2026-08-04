import os
# CUDA 초기화(=torch import) 전에 설정해야 적용됨 — 메모리 파편화로 인한 조기 OOM 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
import re
from tqdm import tqdm
from model import KoJamoNet
from dataset import KoJamoDataset
from config import EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, DATA_DIR, SEQ_LENGTH, STRIDE, BATCH_SIZE

# 매 스텝 동일한 입력 shape(고정 seq_length/batch, drop_last=True)이므로
# cuDNN이 최초 1회만 최적 conv 알고리즘을 탐색하고 이후 캐싱하도록 설정
torch.backends.cudnn.benchmark = True

def save_checkpoint(model, optimizer, scheduler, epoch):
    os.makedirs("checkpoints", exist_ok=True)
    existing = glob.glob("checkpoints/model_v*.pth")
    nums     = [int(re.search(r"model_v(\d+)\.pth", f).group(1))
                for f in existing if re.search(r"model_v(\d+)\.pth", f)]
    save_path = f"checkpoints/model_v{max(nums) + 1 if nums else 1}.pth"
    # torch.compile 대응: 원본 모듈의 state_dict 저장
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # 진짜 이어학습을 위해 모델뿐 아니라 optimizer/scheduler/epoch도 함께 저장
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }, save_path)
    print(f"💾 저장 완료: {save_path} (epoch={epoch})")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 실행 모드 처리 (단순 sys.argv 체크) ---
    # resume: 가중치+optimizer+scheduler+epoch 전부 이어받음 (같은 데이터/설정으로 계속할 때)
    # newstage: 가중치만 이어받고 optimizer/scheduler/epoch은 새로 시작 (Stage1->Stage2처럼
    #           데이터/seq_length가 바뀌는 커리큘럼 전환 시 — 옛 스케줄 꼬리를 물려받지 않기 위함)
    import sys
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    resume_mode = arg == "resume"
    newstage_mode = arg == "newstage"

    print(f"🚀 6트랙 자소 임베딩 + GRU 네트워크 학습 시작. Device: {device}")
    print(f"📁 학습 대상: {DATA_DIR}/ 폴더 내 모든 *.txt 파일")
    print("ℹ️ Ctrl+C 입력 시 안전하게 저장 후 종료됩니다.")

    print(f"📚 데이터: {DATA_DIR} (seq_length={SEQ_LENGTH}, stride={STRIDE})")
    dataset     = KoJamoDataset(data_dir=DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE)
    # drop_last=True: 불균일한 마지막 배치 제거 (cudnn.benchmark가 고정 shape일 때 가장 잘 먹음)
    dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    vocab_sizes = dataset.tokenizer.get_vocab_sizes()

    model     = KoJamoNet(vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
                          num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)

    # resume 시 나중에 optimizer/scheduler 복원용으로 잠시 들고 있을 상태값
    resumed_optimizer_state = None
    resumed_scheduler_state = None
    start_epoch = 1

    if resume_mode or newstage_mode:
        existing = glob.glob("checkpoints/model_v*.pth")
        nums     = [(int(re.search(r"model_v(\d+)\.pth", f).group(1)), f)
                    for f in existing if re.search(r"model_v(\d+)\.pth", f)]
        if nums:
            highest_v, checkpoint_path = max(nums, key=lambda x: x[0])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state_dict)

            if resume_mode and isinstance(checkpoint, dict) and "model" in checkpoint:
                # 신규 포맷 + resume: model/optimizer/scheduler/epoch 전부 이어받음
                resumed_optimizer_state = checkpoint.get("optimizer")
                resumed_scheduler_state = checkpoint.get("scheduler")
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"🔄 {checkpoint_path} — 가중치+optimizer+scheduler 복원, epoch {start_epoch}부터 재개")
            else:
                # newstage 또는 구 포맷: 가중치만 복원, optimizer/scheduler/epoch은 새로 시작
                # (커리큘럼 단계 전환 시 옛 LR 스케줄 꼬리를 물려받지 않기 위함)
                print(f"🔄 {checkpoint_path} — 가중치만 복원, optimizer/LR/epoch은 새로 시작 (새 단계)")
        else:
            print("⚠️ 기존 모델이 없어 처음부터 학습을 시작합니다.")

    # torch.compile 비활성화: GRU+Attention+dropout 조합에서 장시간 학습 시 메모리가
    # 짧은 테스트에서 측정한 값보다 훨씬 크게(batch48 기준 4GB 예상 -> 실제 5.55GB) 누적되어
    # OOM이 재발했음. 원인 특정 실패 — 예측 가능성과 안정성을 위해 컴파일 자체를 끔.
    # (필요시 재활성화: model = torch.compile(model))

    # 성능 개선을 위해 AdamW 사용 및 CosineAnnealingLR 스케줄러 설정
    BASE_LR = 0.003
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)
    # 100 에폭 기준 코사인 어닐링 (학습률이 서서히 0.0001까지 감소하며 정밀 수렴)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

    if resumed_optimizer_state is not None:
        optimizer.load_state_dict(resumed_optimizer_state)
    if resumed_scheduler_state is not None:
        scheduler.load_state_dict(resumed_scheduler_state)

    # LR 워밍업: attention 레이어는 초기화 직후 고LR을 바로 맞으면 가중치가 폭주하기 쉬움
    # (in_proj_weight가 학습 10epoch만에 norm 1000대로 튀는 사고 있었음) — 처음 300 스텝만
    # 0->BASE_LR로 선형 램프업해서 attention이 안정화될 시간을 벌어줌. resume 시엔 생략(이미 지난 단계로 간주).
    WARMUP_STEPS = 300
    global_step = 0
    skip_warmup = resumed_optimizer_state is not None

    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    # bfloat16은 float16보다 표현범위가 넓어(지수부 8비트) 오버플로우로 인한 NaN 발산 위험이 훨씬 낮음.
    # 다이나믹 레인지 문제라 GradScaler(loss scaling)도 필요 없어짐.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    epoch = start_epoch
    try:
         while True:
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                
                # AMP Autocast 적용 (bfloat16 — float16 대비 오버플로우/NaN 발산 위험 낮음)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=device.type == "cuda"):
                    logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num = model(x, target_for_forcing=y)

                    # 정답 y의 타입 감지
                    types_y = model._get_types(y)
                    
                    # --- 6트랙 타입 마스크 Loss 계산 (정적 형상 및 논블로킹 최적화) ---
                    # 1. 타입 분류 손실
                    loss = criterion(logits_type.view(-1, 4), types_y.view(-1))
                    
                    # 2. 각 트랙별 손실 계산 (reduction='none'으로 GPU 동적 마스킹 병목 제거)
                    loss_cho = criterion_none(logits_cho.view(-1, logits_cho.size(-1)), y[:, :, 0].view(-1))
                    loss_jung = criterion_none(logits_jung.view(-1, logits_jung.size(-1)), y[:, :, 1].view(-1))
                    loss_jong = criterion_none(logits_jong.view(-1, logits_jong.size(-1)), y[:, :, 2].view(-1))
                    loss_sym = criterion_none(logits_sym.view(-1, logits_sym.size(-1)), y[:, :, 3].view(-1))
                    loss_eng = criterion_none(logits_eng.view(-1, logits_eng.size(-1)), y[:, :, 4].view(-1))
                    loss_num = criterion_none(logits_num.view(-1, logits_num.size(-1)), y[:, :, 5].view(-1))
                    
                    # 실수형 마스크 생성 (CPU-GPU 동기화 제거)
                    mask_korean = (types_y.view(-1) == 0).float()
                    mask_sym = (types_y.view(-1) == 1).float()
                    mask_eng = (types_y.view(-1) == 2).float()
                    mask_num = (types_y.view(-1) == 3).float()
                    
                    # 마스크 적용 및 평균 계산
                    loss += (loss_cho * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                    loss += (loss_jung * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                    loss += (loss_jong * mask_korean).sum() / (mask_korean.sum() + 1e-8)
                    loss += (loss_sym * mask_sym).sum() / (mask_sym.sum() + 1e-8)
                    loss += (loss_eng * mask_eng).sum() / (mask_eng.sum() + 1e-8)
                    loss += (loss_num * mask_num).sum() / (mask_num.sum() + 1e-8)

                # bfloat16은 loss scaling이 필요 없어 GradScaler 없이 바로 backward
                # 그래도 혹시 모를 NaN/Inf 발산은 여기서 즉시 잡아서 그 스텝만 건너뜀
                # (예전에 이 안전장치 없이 float16+GradScaler로 돌리다 epoch90~100 사이에서
                #  가중치 전체가 NaN으로 발산해 그 뒤 100+ epoch가 통째로 무의미해진 사고가 있었음)
                if not torch.isfinite(loss):
                    print(f"⚠️ 비정상 loss({loss.item()}) 감지 — 이 스텝 건너뜀")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if not skip_warmup and global_step < WARMUP_STEPS:
                    warmup_lr = BASE_LR * (global_step + 1) / WARMUP_STEPS
                    for g in optimizer.param_groups:
                        g['lr'] = warmup_lr
                global_step += 1

                optimizer.step()

                total_loss += loss.item()
                
                # 프로그레스 바에 실시간 loss 및 현재 학습률 표시
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.5f}")

            avg = total_loss / len(dataloader)
            # tqdm 아래에 깔끔하게 최종 에폭 로깅
            print(f"✨ Epoch {epoch} 완료 | Average Loss: {avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.5f}")

            # 10 에폭마다 자동 저장
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch)

            # 스케줄러 갱신
            scheduler.step()
            epoch += 1

    except KeyboardInterrupt:
        print("\n🛑 학습 중단 요청 감지. 현재 가중치를 저장합니다...")
        save_checkpoint(model, optimizer, scheduler, epoch)
        print("👋 종료합니다.")


if __name__ == "__main__":
    train()
