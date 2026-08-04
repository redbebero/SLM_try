# Run status

Verified on 2026-08-04:

- v6 non-raw source data copied to `data/source`; filtered raw KorQuAD expansion
  is in `data/processed/train_expanded.txt`.
- Deduplication and deterministic train/validation split completed by
  `prepare_data.py`.
- Korean Unigram tokenizer trained: 16,384 pieces; training-sample round-trip
  check passes and chat seed has no unknown input tokens.
- Expanded architecture instantiated: 50,340,352 parameters.
- Transformer forward/generation smoke check passes.
- One-batch overfit check passes: loss 5.1369 -> 3.1287.
- The earlier 1,000-step CUDA pretraining run completed on the smaller corpus.
- The 100M-token expansion was prepared and loaded successfully; the 50,000-step
  run was intentionally stopped before completion to keep the usable chat finish
  small and immediate.
- Assistant-only SFT completed for 300 steps on the expanded-vocabulary model:
  loss 9.2712 -> 0.1579.
- Five fixed prompts, including one multi-turn prompt, evaluated in
  `artifacts/evaluation.jsonl`; outputs are coherent Korean and stop at
  `<|end|>`.
- `chat.py` now provides the single interactive entry point with history,
  CUDA/CPU selection, and `/exit`/`/quit` commands.

The distinct prepared corpus contains about 5.1M training tokens, so this run
uses 2M tokens per configured training pass rather than pretending it has
100M–300M distinct tokens. Expanding to that scale requires adding more clean
Korean data or deliberately repeating the corpus.

```bash
PYTHONPATH=. .venv/bin/python chat.py --checkpoint artifacts/chat.pt --prompt '안녕하세요.'
PYTHONPATH=. .venv/bin/python evaluate.py --checkpoint artifacts/chat.pt --tokens 20 --temperature 0.2
```

Raw v6 KorQuAD remains untouched; test/OOD files are not used for training.
