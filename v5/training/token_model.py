"""Small pooled UTF-8 byte model for one-token multi-label classification."""

import torch
from torch import Tensor, nn

from .encoding import PAD_ID

VOCAB_SIZE = 257
TOKEN_MAX_LENGTH = 32
OUTPUT_HEADS = ("attribute_logits", "delta_logits")


class TokenAttributeModel(nn.Module):
    def __init__(self, label_count: int = 0, embedding_dim: int = 32, hidden_dim: int = 48, max_length: int = TOKEN_MAX_LENGTH) -> None:
        super().__init__()
        if label_count <= 0:
            from .token_labels import ATOMIC_LABELS
            label_count = len(ATOMIC_LABELS)
        self.label_count = label_count
        self.max_length = max_length
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.projection = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.attribute_head = nn.Linear(hidden_dim, label_count)
        self.delta_head = nn.Linear(hidden_dim, label_count * 5)

    def forward(self, input_ids: Tensor) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        features = self.embedding(input_ids) + self.position_embedding(positions)
        mask = input_ids.ne(PAD_ID).unsqueeze(-1).to(features.dtype)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        hidden = self.projection(pooled)
        return {
            "attribute_logits": self.attribute_head(hidden),
            "delta_logits": self.delta_head(hidden).view(-1, self.label_count, 5),
        }
