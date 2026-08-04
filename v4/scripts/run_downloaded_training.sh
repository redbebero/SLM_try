#!/usr/bin/env bash
set -euo pipefail

venv/bin/python scripts/train_hrm.py \
  --data-dir experiments/downloaded_sft/train \
  --valid-data-dir experiments/downloaded_sft/valid \
  --init-checkpoint checkpoints/hrm_context_copy_pure_dialogue_v2_12ep_best.pth \
  --output checkpoints/hrm_downloaded_full_epoch1_best.pth \
  --epochs 1 --batch 32 --max-seq-length 256 \
  --emb-dim 16 --hidden-dim 128 --segments 2 --lr 5e-5 \
  --context-encoder --context-layers 1 --copy-head \
  --current-jong --input-dropout 0.05
