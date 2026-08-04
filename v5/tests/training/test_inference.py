import torch

from training.inference import logits_to_proposal


def test_unknown_status_zeroes_semantic_proposal():
    logits = {
        "status": torch.tensor([[4.0, 1.0]]),
        "element": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]]),
        "form": torch.tensor([[0.0, 0.0, 0.0, 0.0, 4.0]]),
        "target": torch.tensor([[0.0, 0.0, 0.0, 4.0]]),
        "power": torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "speed": torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "range": torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "duration": torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    }

    proposal = logits_to_proposal(logits)[0]

    assert proposal["status"] == "UNKNOWN"
    assert proposal["element"] == "UNKNOWN"
    assert proposal["form"] == "UNKNOWN"
    assert proposal["target"] == "UNKNOWN"
    assert proposal["power"] == 0
    assert 0 <= proposal["confidence"] <= 1


def test_proposal_uses_argmax_for_each_head():
    logits = {
        "status": torch.tensor([[0.0, 5.0]]),
        "element": torch.tensor([[0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "form": torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0]]),
        "target": torch.tensor([[0.0, 0.0, 4.0, 0.0]]),
        "power": torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]]),
        "speed": torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        "range": torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0]]),
        "duration": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 6.0]]),
    }

    proposal = logits_to_proposal(logits)[0]

    assert proposal["status"] == "PROPOSAL"
    assert proposal["element"] == "WATER"
    assert proposal["form"] == "SHIELD"
    assert proposal["target"] == "AREA"
    assert proposal["power"] == 2
    assert proposal["duration"] == 5
