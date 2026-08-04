#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs checkpoints
LOG_FILE="logs/unattended_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo " 시작: $(date -Is)"
echo " 로그: $LOG_FILE"

exec 9>/tmp/slm_v3_unattended.lock
if ! flock -n 9; then
    echo "이미 unattended 학습이 실행 중입니다. 종료합니다."
    exit 1
fi

if ! command -v prime-run >/dev/null 2>&1; then
    echo "prime-run을 찾을 수 없습니다."
    exit 1
fi

nvidia-smi >/dev/null
env -u CUDA_DISABLE_PERF_BOOST prime-run venv/bin/python -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA preflight: PASS', torch.cuda.get_device_name(0))"

run_train() {
    echo "\n===== $* ====="
    env -u CUDA_DISABLE_PERF_BOOST prime-run venv/bin/python scripts/train.py "$@"
}

# 1) 빠른 smoke test: 코드·데이터·CUDA·checkpoint 저장 확인
run_train \
    --variant transformer --emb 32 --hidden 256 --layers 2 --heads 4 \
    --batch 64 --seq 128 --stride 128 --epochs 1 --limit-batches 500 \
    --seed 42 --prefix unattended_smoke_

# 2) 같은 조건의 소형 구조 비교
run_train \
    --variant independent --emb 32 --hidden 256 --layers 2 \
    --batch 64 --seq 128 --stride 128 --epochs 3 --limit-batches 1000 \
    --seed 42 --prefix unattended_independent_

run_train \
    --variant cascade --emb 32 --hidden 256 --layers 2 \
    --batch 64 --seq 128 --stride 128 --epochs 3 --limit-batches 1000 \
    --seed 42 --prefix unattended_cascade_

run_train \
    --variant transformer --emb 32 --hidden 256 --layers 2 --heads 4 \
    --batch 64 --seq 128 --stride 128 --epochs 3 --limit-batches 1000 \
    --seed 42 --prefix unattended_transformer_

# 3) 본선 후보 중형 모델 제한 검증
run_train \
    --variant transformer --emb 48 --hidden 384 --layers 4 --heads 6 \
    --batch 64 --seq 128 --stride 128 --epochs 3 --limit-batches 2000 \
    --seed 42 --prefix unattended_medium_

# 4) 위 단계가 모두 정상 종료되면 남은 시간 동안 전체 cleaned corpus 학습.
# EPOCHS를 생략하면 config.py의 EPOCHS=None으로 무한 학습하며,
# 외부 timeout이 SIGINT를 보내면 train.py가 checkpoint를 저장하고 종료한다.
run_train \
    --variant transformer --emb 48 --hidden 384 --layers 4 --heads 6 \
    --batch 64 --seq 250 --stride 250 \
    --seed 42 --prefix unattended_full_

echo " 정상 종료: $(date -Is)"
