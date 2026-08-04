# Korean reasoning evaluation suite

This is a deterministic, evaluation-only suite for the dataset-only Qwen 0.8B upgrade.

- `test.jsonl`: 900 items, 150 per category.
- `ood.jsonl`: 300 items, 50 per category, with separate IDs and templates.
- Categories: arithmetic, multi-step, comparison, state change, temporal logic, and reading inference.

The first five categories have exact numeric or boolean answers. Do not use these files for training or hyperparameter selection.
