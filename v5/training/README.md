# Training

This directory contains only dataset, training, evaluation, and ONNX export code.

Godot is not a dependency of this package.

## Environment

Use Python 3.12:

```bash
uv sync
```

## Planned commands

```bash
uv run python -m training.generate_dataset
uv run python -m training.validate_dataset
uv run python -m training.train
uv run python -m training.evaluate
uv run python -m training.export_onnx

# One-token attribute model
uv run python -m training.generate_token_dataset
uv run python -m training.validate_token_dataset
uv run python -m training.token_train
uv run python -m training.token_export_onnx

# Test one Korean incantation in the terminal
uv run python -m training.token_cli "주변 적에게 붉은 불꽃 구체 빠르게 날려"

# Read the incantation from stdin
echo "빨간 화염 방패" | uv run python -m training.token_cli

# Print JSON for scripts or Godot-side integration tests
uv run python -m training.token_cli --json "붉은 구체"

# Change the attribute confidence threshold
uv run python -m training.token_cli --threshold 0.8 "미지의 주문"
```

`generate_dataset` combines the human-authored base records with controlled wording variants. The validator must pass before training. Model artifacts belong under `models/`.

`evaluate` reports aggregate field/complete accuracy, UNKNOWN detection metrics, per-class precision/recall/F1, and a majority-label baseline calculated from the train split only. A high train score is not sufficient; dev/test complete accuracy and UNKNOWN false-positive rate must be checked.

The one-token experiment uses `data/token-lexicon.json` as its reviewable
source. It generates `data/training-token-attributes.jsonl`, then exports
`models/token_ai.pt` and `models/token_ai.onnx`. The token model has 82 atomic
labels, supports multiple attributes per token, and predicts bounded deltas.
It does not combine tokens or mutate game state.

`training.token_cli` is the terminal verification tool. It splits input only
on whitespace, classifies each Korean token independently, and prints
`kind:value`, `delta`, and confidence. It does not combine attributes or
calculate mana, damage, or game state.
