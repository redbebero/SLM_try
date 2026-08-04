import torch

from training.token_labels import ATOMIC_LABELS
from training.token_model import TokenAttributeModel


def test_token_model_returns_attribute_and_delta_logits():
    model = TokenAttributeModel(embedding_dim=8, hidden_dim=8)
    outputs = model(torch.zeros((2, 32), dtype=torch.long))

    assert tuple(outputs) == ("attribute_logits", "delta_logits")
    assert outputs["attribute_logits"].shape == (2, len(ATOMIC_LABELS))
    assert outputs["delta_logits"].shape == (2, len(ATOMIC_LABELS), 5)
