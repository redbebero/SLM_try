"""Compact GRU encoder-decoder with question-to-answer attention."""

import torch
from torch import nn
import torch.nn.functional as F


class CompactGRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

    def encode(self, source):
        embedded = self.embedding(source)
        outputs, hidden = self.encoder(embedded)
        return outputs, hidden

    def decode_step(self, token, hidden, memory, keys, source_mask):
        if token.ndim == 1:
            token = token.unsqueeze(1)
        query = self.query(hidden[-1]).unsqueeze(1)
        scores = (query * keys).sum(dim=-1)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention.unsqueeze(1), memory)
        embedded = self.embedding(token)
        decoded, hidden = self.decoder(torch.cat([embedded, context], dim=-1), hidden)
        logits = self.output(torch.cat([decoded, context], dim=-1))
        return logits, hidden, attention

    def forward(self, source, decoder_input):
        memory, hidden = self.encode(source)
        keys = self.key(memory)
        source_mask = source.ne(self.pad_id)
        attentions, logits = [], []
        for index in range(decoder_input.size(1)):
            step_logits, hidden, attention = self.decode_step(
                decoder_input[:, index], hidden, memory, keys, source_mask
            )
            logits.append(step_logits)
            attentions.append(attention)
        return torch.cat(logits, dim=1), torch.stack(attentions, dim=1)
