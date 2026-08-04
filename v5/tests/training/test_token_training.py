import torch

from training.token_dataset import records_to_tensors
from training.token_train import attribute_pos_weight, token_multitask_loss
from training.token_model import TokenAttributeModel


def _record(token: str, attributes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "input": {"token": token, "language": "ko"},
        "target": attributes,
    }


def test_token_records_encode_only_token_and_multilabel_targets():
    inputs, targets = records_to_tensors([_record("붉은", [{"kind": "ELEMENT", "value": "FIRE", "delta": 1}])])

    assert inputs.shape == (1, 32)
    assert targets["attributes"].dtype == torch.float32
    assert targets["attributes"].sum().item() == 1
    assert targets["deltas"].dtype == torch.long


def test_token_loss_uses_attribute_and_delta_heads():
    model = TokenAttributeModel(embedding_dim=8, hidden_dim=8)
    logits = model(torch.ones((3, 32), dtype=torch.long))
    _, targets = records_to_tensors([
        _record("붉은", [{"kind": "ELEMENT", "value": "FIRE", "delta": 1}]),
        _record("평범한", []),
        _record("구체", [{"kind": "FORM", "value": "ORB", "delta": 0}]),
    ])

    loss = token_multitask_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_attribute_pos_weight_compensates_for_sparse_multilabel_targets():
    _, targets = records_to_tensors([
        _record("붉은", [{"kind": "ELEMENT", "value": "FIRE", "delta": 1}]),
        _record("평범한", []),
        _record("구체", [{"kind": "FORM", "value": "ORB", "delta": 0}]),
    ])

    weights = attribute_pos_weight(targets)

    assert weights.shape == targets["attributes"].shape[1:]
    assert weights[0].item() == 2
    assert weights[11].item() == 2
