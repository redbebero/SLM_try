"""Compact question/answer dual encoder for train-only answer selection."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class CompactDualEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64, output_dim=64,
                 pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.pad_id = pad_id

    def encode(self, ids):
        if ids.ndim != 2:
            raise ValueError("ids must have shape [batch, length]")
        mask = ids.ne(self.pad_id)
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(ids)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        vector = self.projection(hidden[-1])
        return F.normalize(vector, dim=-1)

    def forward(self, question_ids, answer_ids):
        return self.encode(question_ids), self.encode(answer_ids)
