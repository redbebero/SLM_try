import torch

from training.dataset import records_to_tensors


def test_records_to_tensors_encodes_text_and_all_prediction_heads():
    records = [{
        "input": {"incantation": "붉은 불꽃", "language": "ko"},
        "target": {
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
        },
    }]

    input_ids, targets = records_to_tensors(records)

    assert input_ids.shape == (1, 96)
    assert input_ids.dtype == torch.long
    assert targets["status"].tolist() == [1]
    assert targets["element"].tolist() == [0]
    assert targets["range"].tolist() == [3]
