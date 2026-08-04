import json

import torch

from train_subword_lm import loss_weights, scheduled_inputs, source_balanced_weights


def test_source_balanced_weights_equalize_source_mass():
    rows = [
        {"source": "large"}, {"source": "large"}, {"source": "large"},
        {"source": "small"},
    ]
    weights = source_balanced_weights(rows)
    assert weights[:3] == [1 / 3] * 3
    assert weights[3] == 1.0
    assert sum(weights[:3]) == weights[3]


def test_source_balanced_weights_support_missing_source():
    rows = [{"question": "a"}, {"question": "b"}]
    assert source_balanced_weights(rows) == [0.5, 0.5]


def test_eos_weight_only_changes_eos_target():
    targets = torch.tensor([[4, 3, 5]])
    mask = torch.ones_like(targets, dtype=torch.float32)
    assert torch.allclose(loss_weights(targets, mask, eos_id=3, eos_weight=0.1), torch.tensor([[1.0, 0.1, 1.0]]))


def test_scheduled_inputs_only_replaces_answer_prefix():
    inputs = torch.tensor([[10, 11, 12, 13]])
    predictions = torch.tensor([[90, 91, 92, 93]])
    answer_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    replaced = scheduled_inputs(inputs, predictions, answer_mask, probability=1.0)
    assert replaced.tolist() == [[10, 11, 12, 92]]
