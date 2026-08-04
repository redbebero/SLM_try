import sys

import torch

sys.path.insert(0, "scripts")

from hangul_semantic_generator import HangulSemanticGenerator


def test_semantic_generator_returns_six_track_logits():
    model = HangulSemanticGenerator(emb_dim=8, hidden_dim=16, output_dim=12)
    source = torch.zeros(2, 6, 6, dtype=torch.long)
    source[:, :, 0] = 1
    source[:, :, 1] = 1
    decoder = torch.zeros(2, 5, 6, dtype=torch.long)
    logits = model(source, torch.ones(2, 6, dtype=torch.bool), decoder)

    assert len(logits) == 6
    assert logits[0].shape == (2, 5, 20)
    assert logits[1].shape == (2, 5, 22)
    assert logits[2].shape == (2, 5, 28)


def test_semantic_generator_is_small():
    model = HangulSemanticGenerator(emb_dim=16, hidden_dim=64, output_dim=64)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
