import sys

import torch

sys.path.insert(0, "scripts")

from dual_selector import CompactDualEncoder


def test_dual_encoder_returns_normalized_question_and_answer_vectors():
    model = CompactDualEncoder(vocab_size=64, emb_dim=16, hidden_dim=32, output_dim=24)
    question = torch.randint(1, 64, (3, 7))
    answer = torch.randint(1, 64, (3, 9))

    q, a = model(question, answer)

    assert q.shape == (3, 24)
    assert a.shape == (3, 24)
    assert torch.allclose(q.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(a.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_dual_encoder_is_under_two_million_parameters():
    model = CompactDualEncoder(vocab_size=2048, emb_dim=32, hidden_dim=64, output_dim=64)

    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
