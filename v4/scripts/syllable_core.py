"""Syllable-level GRU with Korean jamo-aware input embeddings."""

from pathlib import Path

import torch
from torch import nn

from tokenizer import KoJamoTokenizer


class SyllableTokenizer:
    def __init__(self):
        jamo = KoJamoTokenizer()
        chars = [chr(code) for code in range(ord("가"), ord("힣") + 1)]
        chars += jamo.sym_list + jamo.eng_list + jamo.num_list + ["?"]
        # Downloaded Korean dialogue contains standalone consonants/vowels
        # (ㅋㅋ, ㅠㅠ, ㄹㅇ). Preserve them instead of collapsing them to UNK.
        chars += jamo.cho_list + jamo.jung_list + jamo.jong_list[1:]
        self.itos = ["<PAD>"] + list(dict.fromkeys(chars))
        self.stoi = {char: index for index, char in enumerate(self.itos)}
        self.jamo = jamo
        self.pad_id = 0
        self.unk_id = self.stoi["?"]

    def encode(self, text):
        return torch.tensor([self.stoi.get(char, self.unk_id) for char in text], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[int(index)] for index in ids if int(index) != self.pad_id)

    def get_vocab_size(self):
        return len(self.itos)

    def jamo_ids(self, ids):
        text = self.decode(ids)
        return self.jamo.encode(text)


class SyllableGRU(nn.Module):
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_cho = nn.Embedding(jamo_vocab_sizes[0], emb_dim)
        self.emb_jung = nn.Embedding(jamo_vocab_sizes[1], emb_dim)
        self.emb_jong = nn.Embedding(jamo_vocab_sizes[2], emb_dim)
        self.proj = nn.Linear(emb_dim * 4, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, jamo_ids):
        jamo = torch.cat([
            self.emb_cho(jamo_ids[:, :, 0]), self.emb_jung(jamo_ids[:, :, 1]),
            self.emb_jong(jamo_ids[:, :, 2]),
        ], dim=-1)
        h = torch.cat([self.embedding(ids), jamo], dim=-1)
        h, _ = self.gru(torch.tanh(self.proj(h)))
        return self.head(h)


class SyllableTransformer(nn.Module):
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_cho = nn.Embedding(jamo_vocab_sizes[0], emb_dim)
        self.emb_jung = nn.Embedding(jamo_vocab_sizes[1], emb_dim)
        self.emb_jong = nn.Embedding(jamo_vocab_sizes[2], emb_dim)
        self.proj = nn.Linear(emb_dim * 4, hidden_dim)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, jamo_ids, causal=True):
        return self.head(self.hidden(ids, jamo_ids, causal=causal))

    def hidden(self, ids, jamo_ids, causal=True):
        jamo = torch.cat([
            self.emb_cho(jamo_ids[:, :, 0]), self.emb_jung(jamo_ids[:, :, 1]),
            self.emb_jong(jamo_ids[:, :, 2]),
        ], dim=-1)
        h = torch.cat([self.embedding(ids), jamo], dim=-1)
        positions = torch.arange(ids.size(1), device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        h = torch.tanh(self.proj(h)) + self.pos(positions)[None, :, :]
        length = ids.size(1)
        mask = torch.triu(torch.ones(length, length, device=ids.device, dtype=torch.bool), diagonal=1) if causal else None
        return self.encoder(h, mask=mask)


class SyllableSpanTransformer(nn.Module):
    """Bidirectional span encoder with explicit context/question segments."""
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_cho = nn.Embedding(jamo_vocab_sizes[0], emb_dim)
        self.emb_jung = nn.Embedding(jamo_vocab_sizes[1], emb_dim)
        self.emb_jong = nn.Embedding(jamo_vocab_sizes[2], emb_dim)
        self.proj = nn.Linear(emb_dim * 4, hidden_dim)
        self.pos = nn.Embedding(max_len, hidden_dim)
        self.segment = nn.Embedding(2, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.start = nn.Linear(hidden_dim, 1)
        self.end = nn.Linear(hidden_dim, 1)

    def forward(self, ids, jamo_ids, segments, padding_mask=None):
        h = self.encode(ids, jamo_ids, segments, padding_mask=padding_mask)
        return self.start(h).squeeze(-1), self.end(h).squeeze(-1)

    def encode(self, ids, jamo_ids, segments, padding_mask=None):
        jamo = torch.cat([
            self.emb_cho(jamo_ids[:, :, 0]), self.emb_jung(jamo_ids[:, :, 1]),
            self.emb_jong(jamo_ids[:, :, 2]),
        ], dim=-1)
        h = torch.cat([self.embedding(ids), jamo], dim=-1)
        positions = torch.arange(ids.size(1), device=ids.device).clamp_max(self.pos.num_embeddings - 1)
        h = torch.tanh(self.proj(h)) + self.pos(positions)[None, :, :] + self.segment(segments)
        return self.encoder(h, mask=None, src_key_padding_mask=padding_mask)


class SyllableCrossSpanTransformer(SyllableSpanTransformer):
    """Small reader that lets each token attend directly to question tokens."""
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__(vocab_size, jamo_vocab_sizes, emb_dim, hidden_dim, layers, heads, max_len)
        self.cross = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(self, ids, jamo_ids, segments, padding_mask=None):
        h = self.encode(ids, jamo_ids, segments, padding_mask=padding_mask)
        question_mask = segments.eq(0)
        if padding_mask is not None:
            question_mask = question_mask | padding_mask
        question_h, _ = self.cross(h, h, h, key_padding_mask=question_mask)
        h = h + question_h
        return self.start(h).squeeze(-1), self.end(h).squeeze(-1)


class SyllablePointerSpanTransformer(SyllableSpanTransformer):
    """Question-conditioned pointer scores for context start/end positions."""
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__(vocab_size, jamo_vocab_sizes, emb_dim, hidden_dim, layers, heads, max_len)
        self.cross = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.start_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.end_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, ids, jamo_ids, segments, padding_mask=None):
        h = self.encode(ids, jamo_ids, segments, padding_mask=padding_mask)
        question_mask = segments.eq(0)
        if padding_mask is not None:
            question_mask = question_mask | padding_mask
        question_h, _ = self.cross(h, h, h, key_padding_mask=question_mask)
        h = h + question_h
        qmask = segments.eq(1)
        if padding_mask is not None:
            qmask = qmask & ~padding_mask
        q = (h * qmask.unsqueeze(-1)).sum(dim=1) / qmask.sum(dim=1, keepdim=True).clamp_min(1)
        q = self.query(q)
        start = (self.start_context(h) * q.unsqueeze(1)).sum(dim=-1) * self.scale
        end = (self.end_context(h) * q.unsqueeze(1)).sum(dim=-1) * self.scale
        return start, end


class SyllableDoubleCrossSpanTransformer(SyllableSpanTransformer):
    """Two lightweight question-to-context attention updates."""
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__(vocab_size, jamo_vocab_sizes, emb_dim, hidden_dim, layers, heads, max_len)
        self.cross1 = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.cross2 = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, ids, jamo_ids, segments, padding_mask=None):
        h = self.encode(ids, jamo_ids, segments, padding_mask=padding_mask)
        question_mask = segments.eq(0)
        if padding_mask is not None:
            question_mask = question_mask | padding_mask
        qh, _ = self.cross1(h, h, h, key_padding_mask=question_mask)
        h = self.norm1(h + qh)
        qh, _ = self.cross2(h, h, h, key_padding_mask=question_mask)
        h = self.norm2(h + qh)
        return self.start(h).squeeze(-1), self.end(h).squeeze(-1)


class SyllableBiaffineSpanTransformer(SyllableCrossSpanTransformer):
    """Cross reader trained to score the complete (start, end) pair."""
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__(vocab_size, jamo_vocab_sizes, emb_dim, hidden_dim, layers, heads, max_len)
        self.start_proj = nn.Linear(hidden_dim, hidden_dim)
        self.end_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear)

    def _hidden(self, ids, jamo_ids, segments, padding_mask=None):
        h = self.encode(ids, jamo_ids, segments, padding_mask=padding_mask)
        question_mask = segments.eq(0)
        if padding_mask is not None:
            question_mask = question_mask | padding_mask
        question_h, _ = self.cross(h, h, h, key_padding_mask=question_mask)
        return h + question_h

    def span_logits(self, ids, jamo_ids, segments, padding_mask=None):
        h = self._hidden(ids, jamo_ids, segments, padding_mask)
        start = self.start_proj(h)
        end = self.end_proj(h)
        return torch.einsum("bih,hk,bjk->bij", start, self.bilinear, end)

    def forward(self, ids, jamo_ids, segments, padding_mask=None):
        scores = self.span_logits(ids, jamo_ids, segments, padding_mask)
        return scores.max(dim=2).values, scores.max(dim=1).values


class SyllableBiDAF(nn.Module):
    def __init__(self, vocab_size, jamo_vocab_sizes, emb_dim=16, hidden_dim=64, layers=1, heads=4, max_len=256):
        super().__init__()
        self.encoder = SyllableSpanTransformer(vocab_size, jamo_vocab_sizes, emb_dim, hidden_dim, layers, heads, max_len)
        self.start = nn.Linear(hidden_dim * 3, 1)
        self.end = nn.Linear(hidden_dim * 3, 1)

    def forward(self, ids, jamo_ids, segments):
        h = self.encoder.encode(ids, jamo_ids, segments)
        qmask = segments.float().unsqueeze(-1)
        q = (h * qmask).sum(dim=1) / qmask.sum(dim=1).clamp_min(1.0)
        fused = torch.cat([h, q[:, None, :].expand_as(h), h * q[:, None, :]], dim=-1)
        return self.start(fused).squeeze(-1), self.end(fused).squeeze(-1)


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
