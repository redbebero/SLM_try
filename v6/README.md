# SLM v6

Goal: fine-tune a small pretrained Qwen model for Korean problem solving and compare it fairly with Korean-from-scratch baselines.

Start with Qwen3.5-0.8B-Base. Follow docs/superpowers/plans/2026-08-03-qwen-korean-problem-solving.md.

## Use the completed Korean adapter

```bash
uv run python scripts/chat_ko.py
```

This loads `Qwen/Qwen3.5-0.8B-Base` plus the completed Korean LoRA adapter at
`models/qwen_0.8b_final_v3`. Type `exit` to quit.
