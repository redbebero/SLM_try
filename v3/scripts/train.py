import os
import argparse
import random
from itertools import islice
# CUDA 초기화(=torch import) 전에 설정해야 적용됨 — 메모리 파편화로 인한 조기 OOM 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# The inherited desktop environment disables NVIDIA CUDA boost globally. Training
# must opt back in; prime-run alone does not remove this variable.
os.environ.pop("CUDA_DISABLE_PERF_BOOST", None)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
import re
from tqdm import tqdm
from model import KoJamoNet, KoJamoTransformer
from dataset import KoJamoDataset
from config import (
    EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, DATA_DIR, SEQ_LENGTH,
    VAL_DATA_DIR, STRIDE, BATCH_SIZE, EPOCHS, BASE_LR, WARMUP_STEPS,
    CHECKPOINT_EVERY_EPOCH, CHECKPOINT_PREFIX, VAL_BATCHES, TEACHER_FORCING_RATIOS,
    JONG_CLASS_WEIGHTS,
    USE_TORCH_COMPILE,
)

# 매 스텝 동일한 입력 shape(고정 seq_length/batch, drop_last=True)이므로
# cuDNN이 최초 1회만 최적 conv 알고리즘을 탐색하고 이후 캐싱하도록 설정
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def training_batch_limit(total_batches, limit_batches):
    """Return the number of batches a limited epoch will actually process."""
    if limit_batches is None:
        return total_batches
    return min(total_batches, max(0, int(limit_batches)))

def save_checkpoint(model, optimizer, scheduler, epoch):
    os.makedirs("checkpoints", exist_ok=True)
    existing = glob.glob(f"checkpoints/{CHECKPOINT_PREFIX}*.pth")
    nums = [int(re.search(rf"{CHECKPOINT_PREFIX}(\d+)\.pth", f).group(1))
            for f in existing if re.search(rf"{CHECKPOINT_PREFIX}(\d+)\.pth", f)]
    save_path = f"checkpoints/{CHECKPOINT_PREFIX}{max(nums) + 1 if nums else 1}.pth"
    # torch.compile 대응: 원본 모듈의 state_dict 저장
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # 진짜 이어학습을 위해 모델뿐 아니라 optimizer/scheduler/epoch도 함께 저장
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
    }, save_path)
    print(f"💾 저장 완료: {save_path} (epoch={epoch})")


def save_best_checkpoint(model, optimizer, scheduler, epoch, metric):
    """Save the best validation checkpoint without affecting numbered saves."""
    os.makedirs("checkpoints", exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_path = f"checkpoints/{CHECKPOINT_PREFIX}best.pth"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": float(metric),
    }, save_path)
    print(f"🏆 Best checkpoint 저장: {save_path} (epoch={epoch}, free={metric:.4f})")


def compute_loss(model, x, y, criterion, criterion_none, use_teacher_forcing,
                 return_outputs=False):
    """Compute active-track loss; optionally use gold jamo for Cascade heads."""
    forcing_ratio = float(use_teacher_forcing)
    forcing_target = y if forcing_ratio > 0.0 else None
    outputs = model(
        x,
        target_for_forcing=forcing_target,
        teacher_forcing_ratio=forcing_ratio,
    )
    logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num = outputs
    types_y = model._get_types(y)

    loss_type = criterion(logits_type.reshape(-1, 4), types_y.reshape(-1))
    jong_weights = torch.as_tensor(
        JONG_CLASS_WEIGHTS, dtype=logits_jong.dtype, device=y.device
    )
    jong_loss = nn.functional.nll_loss(
        logits_jong.reshape(-1, logits_jong.size(-1)),
        y[:, :, 2].reshape(-1),
        weight=jong_weights,
        reduction="none",
    )
    per_track = [
        criterion_none(logits_cho.reshape(-1, logits_cho.size(-1)), y[:, :, 0].reshape(-1)),
        criterion_none(logits_jung.reshape(-1, logits_jung.size(-1)), y[:, :, 1].reshape(-1)),
        jong_loss,
        criterion_none(logits_sym.reshape(-1, logits_sym.size(-1)), y[:, :, 3].reshape(-1)),
        criterion_none(logits_eng.reshape(-1, logits_eng.size(-1)), y[:, :, 4].reshape(-1)),
        criterion_none(logits_num.reshape(-1, logits_num.size(-1)), y[:, :, 5].reshape(-1)),
    ]
    masks = [
        (types_y.reshape(-1) == 0).float(),
        (types_y.reshape(-1) == 0).float(),
        (types_y.reshape(-1) == 0).float(),
        (types_y.reshape(-1) == 1).float(),
        (types_y.reshape(-1) == 2).float(),
        (types_y.reshape(-1) == 3).float(),
    ]
    track_losses = [
        (loss * mask).sum() / (mask.sum() + 1e-8)
        for loss, mask in zip(per_track, masks)
    ]
    total = loss_type + sum(track_losses)
    result = (total, (loss_type, *track_losses))
    if return_outputs:
        return result + ((logits_type, logits_cho, logits_jung, logits_jong,
                          logits_sym, logits_eng, logits_num),)
    return result


@torch.no_grad()
def compute_batch_metrics(model, outputs, y):
    """Return type/jamo accuracy, including final-consonant breakdown."""
    logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num = outputs
    types_y = model._get_types(y)
    predictions = {
        "type": logits_type.argmax(dim=-1),
        "cho": logits_cho.argmax(dim=-1),
        "jung": logits_jung.argmax(dim=-1),
        "jong": logits_jong.argmax(dim=-1),
        "sym": logits_sym.argmax(dim=-1),
        "eng": logits_eng.argmax(dim=-1),
        "num": logits_num.argmax(dim=-1),
    }
    targets = {
        "cho": y[:, :, 0], "jung": y[:, :, 1], "jong": y[:, :, 2],
        "sym": y[:, :, 3], "eng": y[:, :, 4], "num": y[:, :, 5],
    }
    masks = {
        "cho": types_y == 0, "jung": types_y == 0, "jong": types_y == 0,
        "sym": types_y == 1, "eng": types_y == 2, "num": types_y == 3,
    }
    metrics = {
        "type_acc": (predictions["type"] == types_y).float().mean().item()
    }
    for name, mask in masks.items():
        metrics[f"{name}_acc"] = (
            ((predictions[name] == targets[name]) & mask).sum().float()
            / mask.sum().clamp_min(1)
        ).item()

    jong_mask = masks["jong"]
    present = jong_mask & (targets["jong"] > 0)
    empty = jong_mask & (targets["jong"] == 0)
    metrics["jong_present_acc"] = (
        ((predictions["jong"] == targets["jong"]) & present).sum().float()
        / present.sum().clamp_min(1)
    ).item()
    metrics["jong_empty_acc"] = (
        ((predictions["jong"] == targets["jong"]) & empty).sum().float()
        / empty.sum().clamp_min(1)
    ).item()

    hangul = masks["cho"]
    full = (
        (predictions["cho"] == targets["cho"])
        & (predictions["jung"] == targets["jung"])
        & (predictions["jong"] == targets["jong"])
    )
    metrics["full_hangul_acc"] = (full & hangul).sum().float() / hangul.sum().clamp_min(1)
    metrics["full_hangul_acc"] = metrics["full_hangul_acc"].item()
    predicted_hangul = predictions["type"] == 0
    hangul_pairs = predicted_hangul[:, 1:] & predicted_hangul[:, :-1]
    same_previous = (
        hangul_pairs
        & (predictions["cho"][:, 1:] == predictions["cho"][:, :-1])
        & (predictions["jung"][:, 1:] == predictions["jung"][:, :-1])
        & (predictions["jong"][:, 1:] == predictions["jong"][:, :-1])
    )
    metrics["predicted_hangul_repeat_rate"] = (
        same_previous.sum().float() / hangul_pairs.sum().clamp_min(1)
    ).item()
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, criterion, criterion_none, use_teacher_forcing, max_batches):
    model.eval()
    total = 0.0
    count = 0
    metric_totals = {}
    for batch_index, (x, y) in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        loss, _, outputs = compute_loss(
            model, x, y, criterion, criterion_none, use_teacher_forcing,
            return_outputs=True,
        )
        if torch.isfinite(loss):
            total += loss.item()
            count += 1
            for name, value in compute_batch_metrics(model, outputs, y).items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
    model.train()
    average_metrics = {
        name: value / max(1, count) for name, value in metric_totals.items()
    }
    return total / max(1, count), average_metrics


def train():
    global EMB_DIM, HIDDEN_DIM, NUM_LAYERS, EPOCHS, BATCH_SIZE, SEQ_LENGTH, STRIDE
    global CHECKPOINT_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("resume", "newstage"))
    parser.add_argument("--variant", choices=("cascade", "independent", "transformer"), default="cascade")
    parser.add_argument("--emb", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.emb is not None:
        EMB_DIM = args.emb
    if args.hidden is not None:
        HIDDEN_DIM = args.hidden
    if args.layers is not None:
        NUM_LAYERS = args.layers
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.batch is not None:
        BATCH_SIZE = args.batch
    if args.seq is not None:
        SEQ_LENGTH = args.seq
    if args.stride is not None:
        STRIDE = args.stride
    if args.prefix is not None:
        CHECKPOINT_PREFIX = args.prefix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 실행 모드 처리 (단순 sys.argv 체크) ---
    # resume: 가중치+optimizer+scheduler+epoch 전부 이어받음 (같은 데이터/설정으로 계속할 때)
    # newstage: 가중치만 이어받고 optimizer/scheduler/epoch은 새로 시작 (Stage1->Stage2처럼
    #           데이터/seq_length가 바뀌는 커리큘럼 전환 시 — 옛 스케줄 꼬리를 물려받지 않기 위함)
    import sys
    arg = args.mode or ""
    resume_mode = arg == "resume"
    newstage_mode = arg == "newstage"

    print(f"🚀 6트랙 자소 임베딩 네트워크 학습 시작 ({args.variant}). Device: {device}")
    print(f"📁 학습 대상: {DATA_DIR}/ 폴더 내 모든 *.txt 파일")
    print("ℹ️ Ctrl+C 입력 시 안전하게 저장 후 종료됩니다.")

    print(f"📚 데이터: {DATA_DIR} (seq_length={SEQ_LENGTH}, stride={STRIDE})")
    if args.limit_batches is not None:
        print(f"⚡ 제한 실험: epoch당 최대 {args.limit_batches} batch")
    dataset     = KoJamoDataset(data_dir=DATA_DIR, seq_length=SEQ_LENGTH, stride=STRIDE)
    # drop_last=True: 불균일한 마지막 배치 제거 (cudnn.benchmark가 고정 shape일 때 가장 잘 먹음)
    dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    vocab_sizes = dataset.tokenizer.get_vocab_sizes()

    val_dataset = KoJamoDataset(data_dir=VAL_DATA_DIR, seq_length=SEQ_LENGTH, stride=SEQ_LENGTH)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    if args.variant == "transformer":
        model = KoJamoTransformer(
            vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_heads=args.heads, dropout=DROPOUT,
            max_seq_length=max(SEQ_LENGTH, 512),
        ).to(device)
    else:
        model = KoJamoNet(
            vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
            cascade=args.variant == "cascade",
        ).to(device)

    # resume 시 나중에 optimizer/scheduler 복원용으로 잠시 들고 있을 상태값
    resumed_optimizer_state = None
    resumed_scheduler_state = None
    start_epoch = 1

    if resume_mode or newstage_mode:
        existing = glob.glob(f"checkpoints/{CHECKPOINT_PREFIX}*.pth")
        nums     = [(int(re.search(rf"{CHECKPOINT_PREFIX}(\d+)\.pth", f).group(1)), f)
                    for f in existing if re.search(rf"{CHECKPOINT_PREFIX}(\d+)\.pth", f)]
        if nums:
            highest_v, checkpoint_path = max(nums, key=lambda x: x[0])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)

            if resume_mode and isinstance(checkpoint, dict) and "model" in checkpoint:
                # 신규 포맷 + resume: model/optimizer/scheduler/epoch 전부 이어받음
                resumed_optimizer_state = checkpoint.get("optimizer")
                resumed_scheduler_state = checkpoint.get("scheduler")
                saved_epoch = checkpoint.get("epoch", 0)
                saved_scheduler_state = checkpoint.get("scheduler") or {}
                saved_scheduler_step = saved_scheduler_state.get("last_epoch", 0)
                # Ctrl+C 중간 저장은 현재 epoch 번호를 저장하지만 epoch을 끝내지 못함.
                # scheduler step이 해당 epoch의 예상 종료 step보다 작으면 같은 epoch부터
                # 다시 시작해 미완료 epoch을 통째로 건너뛰지 않도록 함.
                if "T_max" in saved_scheduler_state:
                    expected_epoch_end = saved_epoch * len(dataloader) - WARMUP_STEPS
                    start_epoch = (
                        saved_epoch
                        if saved_scheduler_step < expected_epoch_end
                        else saved_epoch + 1
                    )
                else:
                    # ReduceLROnPlateau advances once per completed validation epoch.
                    start_epoch = saved_epoch + 1
                print(f"🔄 {checkpoint_path} — 가중치+optimizer+scheduler 복원, epoch {start_epoch}부터 재개")
            else:
                # newstage 또는 구 포맷: 가중치만 복원, optimizer/scheduler/epoch은 새로 시작
                # (커리큘럼 단계 전환 시 옛 LR 스케줄 꼬리를 물려받지 않기 위함)
                print(f"🔄 {checkpoint_path} — 가중치만 복원, optimizer/LR/epoch은 새로 시작 (새 단계)")
        else:
            print("⚠️ 기존 모델이 없어 처음부터 학습을 시작합니다.")

    if USE_TORCH_COMPILE:
        model = torch.compile(model)

    # 성능 개선을 위해 AdamW 사용. Infinite training uses validation plateau LR.
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
        threshold=1e-3, min_lr=1e-5
    )

    if resumed_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resumed_optimizer_state)
        except Exception as e:
            print(f"⚠️ Optimizer 상태 복원 실패 (파라미터 형태 불일치): {e}")
            print("   -> Optimizer를 신규 초기화하여 학습을 재개합니다.")
            skip_warmup = False
            
    if resumed_scheduler_state is not None:
        try:
            scheduler.load_state_dict(resumed_scheduler_state)
        except Exception as e:
            print(f"⚠️ Scheduler 상태 복원 실패: {e}")
            print("   -> Scheduler를 신규 초기화합니다.")

    # LR 워밍업: attention 레이어는 초기화 직후 고LR을 바로 맞으면 가중치가 폭주하기 쉬움
    # (in_proj_weight가 학습 10epoch만에 norm 1000대로 튀는 사고 있었음) — 처음 300 스텝만
    # 0->BASE_LR로 선형 램프업해서 attention이 안정화될 시간을 벌어줌. resume 시엔 생략(이미 지난 단계로 간주).
    global_step = 0
    skip_warmup = resumed_optimizer_state is not None

    criterion = nn.NLLLoss()
    criterion_none = nn.NLLLoss(reduction='none')

    # bfloat16은 float16보다 표현범위가 넓어(지수부 8비트) 오버플로우로 인한 NaN 발산 위험이 훨씬 낮음.
    # 다이나믹 레인지 문제라 GradScaler(loss scaling)도 필요 없어짐.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    epoch = start_epoch
    best_val_free = float("inf")
    try:
         while EPOCHS is None or epoch <= EPOCHS:
            total_loss = 0
            component_totals = {name: 0.0 for name in (
                "type", "cho", "jung", "jong", "sym", "eng", "num"
            )}
            component_counts = {name: 0 for name in component_totals}
            ratio_index = min(epoch - 1, len(TEACHER_FORCING_RATIOS) - 1)
            teacher_forcing_ratio = TEACHER_FORCING_RATIOS[ratio_index]
            print(f"🧭 Epoch {epoch}: teacher_forcing_ratio={teacher_forcing_ratio:.2f}")
            epoch_batches = training_batch_limit(len(dataloader), args.limit_batches)
            epoch_loader = islice(dataloader, epoch_batches)
            pbar = tqdm(epoch_loader, total=epoch_batches, desc=f"Epoch {epoch}", leave=True)
            processed_batches = 0
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=device.type == "cuda"):
                    loss, component_losses = compute_loss(
                        model, x, y, criterion, criterion_none, teacher_forcing_ratio
                    )

                    for name, value in (
                        ("type", component_losses[0]), ("cho", component_losses[1]),
                        ("jung", component_losses[2]), ("jong", component_losses[3]),
                        ("sym", component_losses[4]), ("eng", component_losses[5]),
                        ("num", component_losses[6]),
                    ):
                        component_totals[name] += float(value.detach())
                        component_counts[name] += 1

                # bfloat16은 loss scaling이 필요 없어 GradScaler 없이 바로 backward
                # 그래도 혹시 모를 NaN/Inf 발산은 여기서 즉시 잡아서 그 스텝만 건너뜀
                # (예전에 이 안전장치 없이 float16+GradScaler로 돌리다 epoch90~100 사이에서
                #  가중치 전체가 NaN으로 발산해 그 뒤 100+ epoch가 통째로 무의미해진 사고가 있었음)
                if not torch.isfinite(loss):
                    print(f"⚠️ 비정상 loss({loss.item()}) 감지 — 이 스텝 건너뜀")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if not skip_warmup and global_step < WARMUP_STEPS:
                    warmup_lr = BASE_LR * (global_step + 1) / WARMUP_STEPS
                    for g in optimizer.param_groups:
                        g['lr'] = warmup_lr
                global_step += 1

                optimizer.step()
                total_loss += loss.item()
                processed_batches += 1
                
                # 프로그레스 바에 실시간 loss 및 현재 학습률 표시
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.5f}")

            avg = total_loss / max(1, processed_batches)
            # tqdm 아래에 깔끔하게 최종 에폭 로깅
            component_avg = " ".join(
                f"{name}={component_totals[name] / max(1, component_counts[name]):.3f}"
                for name in component_totals
            )
            print(
                f"✨ Epoch {epoch} 완료 | Average Loss: {avg:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | {component_avg}"
            )

            val_teacher, teacher_metrics = evaluate(
                model, val_dataloader, device, criterion, criterion_none,
                use_teacher_forcing=True, max_batches=VAL_BATCHES
            )
            val_free, free_metrics = evaluate(
                model, val_dataloader, device, criterion, criterion_none,
                use_teacher_forcing=False, max_batches=VAL_BATCHES
            )
            print(
                f"📊 Validation | teacher={val_teacher:.4f} | free_running={val_free:.4f} | "
                f"free_type={free_metrics.get('type_acc', 0.0):.3f} "
                f"free_cho={free_metrics.get('cho_acc', 0.0):.3f} "
                f"free_jung={free_metrics.get('jung_acc', 0.0):.3f} "
                f"free_jong={free_metrics.get('jong_acc', 0.0):.3f} "
                f"jong+={free_metrics.get('jong_present_acc', 0.0):.3f} "
                f"full_hangul={free_metrics.get('full_hangul_acc', 0.0):.3f} "
                f"repeat={free_metrics.get('predicted_hangul_repeat_rate', 0.0):.3f}"
            )
            scheduler.step(val_free)

            if val_free < best_val_free:
                best_val_free = val_free
                save_best_checkpoint(model, optimizer, scheduler, epoch, val_free)

            # 10 에폭마다 자동 저장
            if epoch % CHECKPOINT_EVERY_EPOCH == 0:
                save_checkpoint(model, optimizer, scheduler, epoch)

            # 스케줄러 갱신 (배치 루프 안으로 이동됨)
            epoch += 1
         
         if EPOCHS is not None:
             print(f"🎉 지정된 {EPOCHS}에폭 학습이 정상 완료되었습니다! 최종 체크포인트를 저장합니다...")
             save_checkpoint(model, optimizer, scheduler, epoch - 1)

    except KeyboardInterrupt:
        print("\n🛑 학습 중단 요청 감지. 현재 가중치를 저장합니다...")
        save_checkpoint(model, optimizer, scheduler, epoch)
        print("👋 종료합니다.")


if __name__ == "__main__":
    train()
