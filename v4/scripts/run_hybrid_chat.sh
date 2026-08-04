#!/usr/bin/env bash
set -euo pipefail

exec venv/bin/python scripts/chat.py \
  checkpoints/hrm_context_copy_pure_dialogue_v2_12ep_best.pth \
  --intent-checkpoint checkpoints/hrm_intent_pure_v3_best.pth \
  --reasoning-checkpoint checkpoints/hrm_context_reasoning_order_finetune_2ep_best.pth \
  "$@"
