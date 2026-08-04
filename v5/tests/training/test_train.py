import torch

from training.model import OUTPUT_HEADS, SpellProposalModel
from training.train import multitask_loss


def test_multitask_loss_uses_every_prediction_head():
    model = SpellProposalModel(embedding_dim=8, hidden_dim=8)
    logits = model(torch.ones((2, 96), dtype=torch.long))
    targets = {name: torch.zeros(2, dtype=torch.long) for name in OUTPUT_HEADS}

    loss = multitask_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0
