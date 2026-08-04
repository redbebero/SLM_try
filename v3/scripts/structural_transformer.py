"""Compact Korean-jamo causal Transformer with a joint Hangul output head."""

import torch
import torch.nn as nn


class StructuralKoreanTransformer(nn.Module):
    def __init__(self, vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=32,
                 hidden_dim=256, layers=3, heads=4, max_seq_length=256):
        super().__init__()
        n_cho, n_jung, n_jong, n_sym, n_eng, n_num = vocab_sizes
        self.vocab_sizes = vocab_sizes
        self.max_seq_length = max_seq_length
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, emb_dim, padding_idx=0)
            for size in vocab_sizes
        ])
        self.type_emb = nn.Embedding(4, emb_dim)
        self.input_proj = nn.Linear(emb_dim * 7, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.type_head = nn.Linear(hidden_dim, 4)
        self.char_head = nn.Linear(hidden_dim, 19 * 21 * 28)
        self.sym_head = nn.Linear(hidden_dim, n_sym)
        self.eng_head = nn.Linear(hidden_dim, n_eng)
        self.num_head = nn.Linear(hidden_dim, n_num)

    @staticmethod
    def token_types(x):
        return ((x[:, :, 3] > 0).long()
                + 2 * (x[:, :, 4] > 0).long()
                + 3 * (x[:, :, 5] > 0).long())

    def forward(self, x):
        types = self.token_types(x)
        previous_jong = torch.cat([
            torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device),
            x[:, :-1, 2],
        ], dim=1)
        tracks = [x[:, :, 0], x[:, :, 1], previous_jong,
                  x[:, :, 3], x[:, :, 4], x[:, :, 5]]
        embedded = [embedding(track) for embedding, track in zip(self.embeddings, tracks)]
        embedded.append(self.type_emb(types))
        h = self.input_proj(torch.cat(embedded, dim=-1))
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = h + self.pos_emb(positions)
        causal = torch.triu(
            torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.norm(self.encoder(h, mask=causal))
        return self.type_head(h), self.char_head(h), self.sym_head(h), self.eng_head(h), self.num_head(h)


def load_structural_checkpoint(path, vocab_sizes, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = StructuralKoreanTransformer(vocab_sizes=vocab_sizes)
    model.load_state_dict(state)
    return model.to(device).eval()
