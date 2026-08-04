import sys

import torch

sys.path.insert(0, "scripts")

from subword_lm import CompactSubwordCausalLM
from train_subword_lm import SubwordTokenizer


def test_compact_subword_decoder_is_causal_and_small():
    model = CompactSubwordCausalLM(
        vocab_size=2048, emb_dim=16, hidden_dim=64,
        layers=1, heads=4, max_len=128,
    ).eval()
    ids = torch.randint(0, 2048, (1, 8))
    with torch.no_grad():
        logits = model(ids)
        changed = ids.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 2048
        changed_logits = model(changed)
    assert logits.shape == (1, 8, 2048)
    assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_subword_decode_removes_control_tokens_from_answer_text():
    tokenizer = SubwordTokenizer("experiments/subword_tokenizer_chat/ko.model")

    decoded = tokenizer.decode(tokenizer.encode("답변<eos>"))

    assert decoded == "답변"
