"""Small answer decoder conditioned on a Hangul semantic vector."""

import torch
from torch import nn

from hangul_semantic_encoder import HangulSemanticEncoder, VOCAB_SIZES


class HangulSemanticGenerator(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=64, output_dim=64,
                 vocab_sizes=VOCAB_SIZES):
        super().__init__()
        self.vocab_sizes = tuple(vocab_sizes)
        self.semantic_encoder = HangulSemanticEncoder(
            num_categories=6, emb_dim=emb_dim, hidden_dim=hidden_dim,
            output_dim=output_dim, vocab_sizes=vocab_sizes,
        )
        self.decoder_embeddings = nn.ModuleList([
            nn.Embedding(size, emb_dim, padding_idx=0) for size in vocab_sizes
        ])
        self.decoder_type = nn.Embedding(4, emb_dim)
        self.decoder_input = nn.Linear(emb_dim * 7, hidden_dim)
        self.initial = nn.Linear(output_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, size) for size in vocab_sizes
        ])

    @staticmethod
    def _types(x):
        return ((x[:, :, 3] > 0).long()
                + 2 * (x[:, :, 4] > 0).long()
                + 3 * (x[:, :, 5] > 0).long()).clamp_max(3)

    def _decode_input(self, decoder_input):
        tracks = [embedding(decoder_input[:, :, index])
                  for index, embedding in enumerate(self.decoder_embeddings)]
        tracks.append(self.decoder_type(self._types(decoder_input)))
        return self.decoder_input(torch.cat(tracks, dim=-1))

    def forward(self, source, source_mask, decoder_input):
        vector, _ = self.semantic_encoder(source, source_mask)
        hidden = self.initial(vector).unsqueeze(0)
        decoded, _ = self.decoder(self._decode_input(decoder_input), hidden)
        return tuple(head(decoded) for head in self.heads)


def load_semantic_encoder(model, checkpoint):
    """Copy a trained semantic encoder checkpoint into a generator."""
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    encoder_state = {
        key[len("semantic_encoder."):]: value
        for key, value in state.items() if key.startswith("semantic_encoder.")
    }
    if not encoder_state:
        encoder_state = state
    model.semantic_encoder.load_state_dict(encoder_state, strict=False)
