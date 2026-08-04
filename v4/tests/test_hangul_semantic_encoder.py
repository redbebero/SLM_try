import sys

import torch

sys.path.insert(0, "scripts")

from hangul_semantic_encoder import HangulSemanticEncoder


def test_encoder_returns_normalized_sentence_vectors_and_category_logits():
    model = HangulSemanticEncoder(num_categories=6, emb_dim=8, hidden_dim=16,
                                  output_dim=12)
    x = torch.zeros(2, 7, 6, dtype=torch.long)
    x[:, :, 0] = 1
    x[:, :, 1] = 1
    mask = torch.ones(2, 7, dtype=torch.bool)

    vectors, logits = model(x, mask)

    assert vectors.shape == (2, 12)
    assert logits.shape == (2, 6)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_encoder_has_jamo_reconstruction_heads():
    model = HangulSemanticEncoder(num_categories=6, emb_dim=8, hidden_dim=16,
                                  output_dim=12)
    x = torch.zeros(2, 5, 6, dtype=torch.long)
    x[:, :, 0] = 1
    x[:, :, 1] = 1
    outputs = model.reconstruct(x, torch.ones(2, 5, dtype=torch.bool))

    assert len(outputs) == 6
    assert outputs[0].shape == (2, 5, 20)
    assert outputs[1].shape == (2, 5, 22)
    assert outputs[2].shape == (2, 5, 28)
