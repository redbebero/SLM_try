"""Small causal LM over a learned Korean subword vocabulary."""

import torch
from torch import nn


class CompactSubwordCausalLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=64,
                 layers=1, heads=4, max_len=256):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, ids):
        if ids.ndim != 2:
            raise ValueError("ids must have shape [batch, length]")
        length = ids.size(1)
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        positions = torch.arange(length, device=ids.device).unsqueeze(0)
        hidden = self.input_proj(self.embedding(ids)) + self.pos(positions)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1)
        hidden = self.encoder(hidden, mask=mask)
        return self.head(self.norm(hidden))
