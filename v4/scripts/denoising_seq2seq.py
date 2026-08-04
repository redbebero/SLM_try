"""Small denoising seq2seq utilities for Korean text pretraining."""

import torch


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def corrupt_tokens(tokens, mask_prob=0.15, mask_id=1):
    """Replace sampled input tokens with SentencePiece's unknown token."""
    result = tokens.clone()
    selected = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    result[selected] = mask_id
    return result
