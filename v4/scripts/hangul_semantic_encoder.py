"""Compact Hangul-factorized sentence encoder.

It shares one encoder between questions and answers.  Each Hangul syllable is
represented by its six tokenizer tracks, while the auxiliary heads make the
intermediate representation testable instead of relying only on generation.
"""

import torch
from torch import nn
import torch.nn.functional as F


VOCAB_SIZES = (20, 22, 28, 36, 53, 11)


class HangulSemanticEncoder(nn.Module):
    def __init__(self, num_categories, emb_dim=16, hidden_dim=64, output_dim=64,
                 vocab_sizes=VOCAB_SIZES):
        super().__init__()
        self.vocab_sizes = tuple(vocab_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, emb_dim, padding_idx=0)
            for size in self.vocab_sizes
        ])
        self.type_embedding = nn.Embedding(4, emb_dim)
        self.input_projection = nn.Linear(emb_dim * 7, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        self.category_head = nn.Linear(output_dim, num_categories)
        self.reconstruction_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 2, size) for size in self.vocab_sizes
        ])

    @staticmethod
    def _types(x):
        return ((x[:, :, 3] > 0).long()
                + 2 * (x[:, :, 4] > 0).long()
                + 3 * (x[:, :, 5] > 0).long()).clamp_max(3)

    def _hidden(self, x):
        tracks = [embedding(x[:, :, index])
                  for index, embedding in enumerate(self.embeddings)]
        tracks.append(self.type_embedding(self._types(x)))
        hidden, _ = self.encoder(self.input_projection(torch.cat(tracks, dim=-1)))
        return hidden

    def forward(self, x, mask=None):
        hidden = self._hidden(x)
        if mask is None:
            mask = x.ne(0).any(dim=-1)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        vector = F.normalize(self.projection(pooled), dim=-1)
        return vector, self.category_head(vector)

    def reconstruct(self, x, mask=None):
        hidden = self._hidden(x)
        return tuple(head(hidden) for head in self.reconstruction_heads)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
