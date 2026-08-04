"""Small tokenizer-free multi-task classifier for Korean incantations."""

import torch
from torch import Tensor, nn

from .encoding import MAX_LENGTH, PAD_ID

VOCAB_SIZE = 257
OUTPUT_HEADS = (
    "status",
    "element",
    "form",
    "target",
    "power",
    "speed",
    "range",
    "duration",
)

HEAD_SIZES = {
    "status": 2,
    "element": 8,
    "form": 5,
    "target": 4,
    "power": 6,
    "speed": 6,
    "range": 6,
    "duration": 6,
}


class SpellProposalModel(nn.Module):
    """Predict fixed spell-semantic labels from shifted UTF-8 byte IDs."""

    def __init__(
        self,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.output_heads = OUTPUT_HEADS
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, HEAD_SIZES[name]) for name in OUTPUT_HEADS
        })

    def forward(self, input_ids: Tensor) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        token_features = self.embedding(input_ids) + self.position_embedding(positions)
        mask = input_ids.ne(PAD_ID).unsqueeze(-1).to(token_features.dtype)
        pooled = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        hidden = self.projection(pooled)
        return {name: self.heads[name](hidden) for name in OUTPUT_HEADS}
