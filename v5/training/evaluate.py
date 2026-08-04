"""Evaluate held-out incantations without using game-engine state."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .data import load_records, split_records
from .dataset import records_to_tensors
from .inference import logits_to_proposal
from .labels import LABELS_BY_FIELD
from .model import SpellProposalModel

EVALUATED_FIELDS = ("status", "element", "form", "target", "power", "speed", "range", "duration")
CATEGORICAL_FIELDS = tuple(LABELS_BY_FIELD)


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _per_class_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in CATEGORICAL_FIELDS:
        field_result: dict[str, dict[str, float | int]] = {}
        for label in LABELS_BY_FIELD[field]:
            true_positive = sum(
                prediction[field] == label and target[field] == label
                for prediction, target in zip(predictions, targets)
            )
            predicted_count = sum(prediction[field] == label for prediction in predictions)
            support = sum(target[field] == label for target in targets)
            precision = _divide(true_positive, predicted_count)
            recall = _divide(true_positive, support)
            field_result[label] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": _divide(2 * precision * recall, precision + recall)
                if precision + recall
                else 0.0,
            }
        result[field] = field_result
    return result


def _unknown_detection_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, float]:
    predicted_unknown = [prediction["status"] == "UNKNOWN" for prediction in predictions]
    actual_unknown = [target["status"] == "UNKNOWN" for target in targets]
    true_positive = sum(predicted and actual for predicted, actual in zip(predicted_unknown, actual_unknown))
    false_positive = sum(predicted and not actual for predicted, actual in zip(predicted_unknown, actual_unknown))
    false_negative = sum(not predicted and actual for predicted, actual in zip(predicted_unknown, actual_unknown))
    true_negative = sum(not predicted and not actual for predicted, actual in zip(predicted_unknown, actual_unknown))
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    return {
        "accuracy": _divide(true_positive + true_negative, len(predictions)),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall)
        if precision + recall
        else 0.0,
        "false_positive_rate": _divide(false_positive, false_positive + true_negative),
    }


def prediction_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must have equal non-zero length")

    correct_fields = sum(
        prediction[field] == target[field]
        for prediction, target in zip(predictions, targets)
        for field in EVALUATED_FIELDS
    )
    total_fields = len(predictions) * len(EVALUATED_FIELDS)
    complete = sum(
        all(prediction[field] == target[field] for field in EVALUATED_FIELDS)
        for prediction, target in zip(predictions, targets)
    )
    return {
        "field_accuracy": correct_fields / total_fields,
        "complete_accuracy": complete / len(predictions),
        "mean_confidence": sum(float(prediction["confidence"]) for prediction in predictions) / len(predictions),
        "unknown_detection": _unknown_detection_metrics(predictions, targets),
        "per_class": _per_class_metrics(predictions, targets),
    }


def _majority_value(values: list[Any], labels: tuple[Any, ...]) -> Any:
    counts = Counter(values)
    return max(labels, key=lambda label: (counts[label], -labels.index(label)))


def majority_baseline_metrics(
    training_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a deterministic majority-label baseline from training data only."""
    if not training_records or not evaluation_records:
        raise ValueError("training and evaluation records must be non-empty")

    training_targets = [record["target"] for record in training_records]
    majority = {
        field: _majority_value(
            [target[field] for target in training_targets],
            LABELS_BY_FIELD[field] if field in LABELS_BY_FIELD else tuple(range(6)),
        )
        for field in EVALUATED_FIELDS
    }
    if majority["status"] == "UNKNOWN":
        majority.update({"element": "UNKNOWN", "form": "UNKNOWN", "target": "UNKNOWN"})
        majority.update({field: 0 for field in ("power", "speed", "range", "duration")})

    predictions = [
        {
            "schema_version": 1,
            **majority,
            "confidence": 0.0,
        }
        for _ in evaluation_records
    ]
    return prediction_metrics(predictions, [record["target"] for record in evaluation_records])


def evaluate_model(model: SpellProposalModel, records: list[dict[str, Any]]) -> dict[str, float]:
    input_ids, _ = records_to_tensors(records)
    model.eval()
    with torch.no_grad():
        predictions = logits_to_proposal(model(input_ids))
    targets = [record["target"] for record in records]
    return prediction_metrics(predictions, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/training-spells.expanded.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/spell_ai.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpellProposalModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    splits = split_records(load_records(args.data))
    results = {split: evaluate_model(model, records) for split, records in splits.items() if records}
    for split in ("dev", "test"):
        if splits[split]:
            results[f"{split}_majority_baseline"] = majority_baseline_metrics(splits["train"], splits[split])
    print(results)


if __name__ == "__main__":
    main()
