"""Compact causal language model with one target per Korean syllable."""

import torch
from torch import nn


class FullSyllableCausalLM(nn.Module):
    """Question-conditioned causal decoder over complete Korean characters."""

    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16,
                 hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_cho = nn.Embedding(jamo_vocab_sizes[0], emb_dim, padding_idx=0)
        self.emb_jung = nn.Embedding(jamo_vocab_sizes[1], emb_dim, padding_idx=0)
        self.emb_jong = nn.Embedding(jamo_vocab_sizes[2], emb_dim, padding_idx=0)
        self.input_proj = nn.Linear(emb_dim * 4, hidden_dim)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, ids, jamo_ids):
        if ids.ndim != 2 or jamo_ids.ndim != 3:
            raise ValueError("ids must be [batch, length], jamo_ids [batch, length, 6]")
        length = ids.size(1)
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        jamo = torch.cat([
            self.emb_cho(jamo_ids[:, :, 0]),
            self.emb_jung(jamo_ids[:, :, 1]),
            self.emb_jong(jamo_ids[:, :, 2]),
        ], dim=-1)
        hidden = self.input_proj(torch.cat([self.embedding(ids), jamo], dim=-1))
        position = torch.arange(length, device=ids.device).unsqueeze(0)
        hidden = hidden + self.pos(position)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.head(self.norm(hidden))
