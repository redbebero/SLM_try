import pytest

from training.evaluate import majority_baseline_metrics, prediction_metrics


def test_prediction_metrics_reports_field_and_complete_accuracy():
    target = {
        "schema_version": 1,
        "status": "PROPOSAL",
        "element": "FIRE",
        "form": "ORB",
        "target": "ENEMY",
        "power": 2,
        "speed": 1,
        "range": 3,
        "duration": 0,
        "confidence": 1,
    }
    prediction = {**target, "confidence": 0.8}

    metrics = prediction_metrics([prediction], [target])

    assert metrics["complete_accuracy"] == 1.0
    assert metrics["field_accuracy"] == 1.0
    assert metrics["mean_confidence"] == 0.8


def test_prediction_metrics_reports_unknown_detection_and_per_class_scores():
    proposal_target = {
        "schema_version": 1,
        "status": "PROPOSAL",
        "element": "FIRE",
        "form": "ORB",
        "target": "ENEMY",
        "power": 2,
        "speed": 1,
        "range": 3,
        "duration": 0,
        "confidence": 1,
    }
    unknown_target = {
        "schema_version": 1,
        "status": "UNKNOWN",
        "element": "UNKNOWN",
        "form": "UNKNOWN",
        "target": "UNKNOWN",
        "power": 0,
        "speed": 0,
        "range": 0,
        "duration": 0,
        "confidence": 1,
    }

    metrics = prediction_metrics(
        [
            {**proposal_target, "confidence": 0.8},
            {**unknown_target, "confidence": 0.6},
        ],
        [proposal_target, unknown_target],
    )

    assert metrics["unknown_detection"] == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "false_positive_rate": 0.0,
    }
    assert metrics["per_class"]["element"]["FIRE"] == {
        "support": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert metrics["per_class"]["element"]["UNKNOWN"]["support"] == 1


def test_majority_baseline_uses_training_records_only():
    train_target = {
        "schema_version": 1,
        "status": "PROPOSAL",
        "element": "FIRE",
        "form": "ORB",
        "target": "ENEMY",
        "power": 2,
        "speed": 1,
        "range": 3,
        "duration": 0,
        "confidence": 1,
    }
    eval_target = {**train_target, "element": "WATER", "confidence": 1}
    train_records = [{"target": train_target}, {"target": train_target}]
    eval_records = [{"target": eval_target}]

    metrics = majority_baseline_metrics(train_records, eval_records)

    assert metrics["field_accuracy"] == pytest.approx(7 / 8)
    assert metrics["complete_accuracy"] == 0.0
