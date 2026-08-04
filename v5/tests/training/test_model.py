import torch

from training.model import OUTPUT_HEADS, SpellProposalModel


def test_model_returns_fixed_multitask_output_heads():
    model = SpellProposalModel(embedding_dim=16, hidden_dim=16)
    outputs = model(torch.zeros((2, 96), dtype=torch.long))

    assert tuple(outputs) == OUTPUT_HEADS
    assert outputs["status"].shape == (2, 2)
    assert outputs["element"].shape == (2, 8)
    assert outputs["form"].shape == (2, 5)
    assert outputs["target"].shape == (2, 4)
    assert outputs["power"].shape == (2, 6)
    assert outputs["speed"].shape == (2, 6)
    assert outputs["range"].shape == (2, 6)
    assert outputs["duration"].shape == (2, 6)


def test_model_has_no_game_state_output_heads():
    model = SpellProposalModel()

    assert not {"hp", "mana", "damage", "position"}.intersection(model.output_heads)
