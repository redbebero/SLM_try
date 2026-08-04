import sys

import torch

sys.path.insert(0, "scripts")

from conditioned_dual_selector import ConditionedDualEncoder


def test_conditioned_dual_encoder_returns_act_logits():
    model = ConditionedDualEncoder(64, num_acts=4, emb_dim=16, hidden_dim=32,
                                   output_dim=24)
    q = torch.randint(1, 64, (3, 7))
    a = torch.randint(1, 64, (3, 8))
    qv, av, qlogits, alogits = model.forward_with_act(q, a)
    assert qv.shape == av.shape == (3, 24)
    assert qlogits.shape == alogits.shape == (3, 4)


def test_conditioned_dual_encoder_is_small():
    model = ConditionedDualEncoder(2048, num_acts=8, emb_dim=32,
                                   hidden_dim=64, output_dim=64)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
