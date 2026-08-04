import sys

import torch

sys.path.insert(0, "scripts")

from denoising_seq2seq import corrupt_tokens, count_parameters
from seq2seq_lm import CompactSeq2SeqLM


def test_corruption_preserves_length_and_changes_selected_tokens():
    tokens = torch.tensor([4, 5, 6, 7, 8])
    corrupted = corrupt_tokens(tokens, mask_prob=1.0, mask_id=1)
    assert corrupted.shape == tokens.shape
    assert torch.all(corrupted == 1)


def test_denoising_model_is_under_two_million_parameters():
    model = CompactSeq2SeqLM(2048, emb_dim=16, hidden_dim=64, layers=1, heads=4)
    assert count_parameters(model) < 2_000_000
