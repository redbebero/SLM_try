"""Compact question-encoder/answer-decoder Transformer."""

import torch
from torch import nn


class CompactSeq2SeqLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=64, layers=1,
                 heads=4, max_src_len=256, max_tgt_len=256, pad_id=0):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.src_pos = nn.Embedding(max_src_len, hidden_dim)
        self.tgt_pos = nn.Embedding(max_tgt_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id

    def forward(self, source, decoder_input):
        src_len, tgt_len = source.size(1), decoder_input.size(1)
        if src_len > self.max_src_len or tgt_len > self.max_tgt_len:
            raise ValueError("sequence exceeds configured length")
        src_pos = torch.arange(src_len, device=source.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt_len, device=decoder_input.device).unsqueeze(0)
        src = self.input_proj(self.embedding(source)) + self.src_pos(src_pos)
        tgt = self.input_proj(self.embedding(decoder_input)) + self.tgt_pos(tgt_pos)
        memory = self.encoder(src, src_key_padding_mask=source.eq(self.pad_id))
        causal = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device), diagonal=1)
        decoded = self.decoder(
            tgt, memory, tgt_mask=causal,
            tgt_key_padding_mask=decoder_input.eq(self.pad_id),
            memory_key_padding_mask=source.eq(self.pad_id),
        )
        return self.head(self.norm(decoded))
