import sys

import torch

sys.path.insert(0, "scripts")

from gru_seq2seq import CompactGRUSeq2Seq
from train_gru_seq2seq import scheduled_decoder_inputs


def test_gru_seq2seq_returns_answer_logits_and_attention():
    model = CompactGRUSeq2Seq(vocab_size=128, emb_dim=16, hidden_dim=32, pad_id=0)
    source = torch.randint(1, 128, (2, 7))
    decoder_input = torch.randint(1, 128, (2, 5))

    logits, attention = model(source, decoder_input)

    assert logits.shape == (2, 5, 128)
    assert attention.shape == (2, 5, 7)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_gru_seq2seq_is_small():
    model = CompactGRUSeq2Seq(vocab_size=2048, emb_dim=32, hidden_dim=64, pad_id=0)

    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_decode_step_matches_batched_forward_logits():
    torch.manual_seed(4)
    model = CompactGRUSeq2Seq(vocab_size=128, emb_dim=16, hidden_dim=32, pad_id=0).eval()
    source = torch.randint(1, 128, (1, 7))
    decoder_input = torch.randint(1, 128, (1, 5))
    with torch.no_grad():
        expected, _ = model(source, decoder_input)
        memory, hidden = model.encode(source)
        keys = model.key(memory)
        actual = []
        for index in range(decoder_input.size(1)):
            step, hidden, _ = model.decode_step(
                decoder_input[:, index], hidden, memory, keys, source.ne(model.pad_id)
            )
            actual.append(step)
    assert torch.allclose(expected, torch.cat(actual, dim=1), atol=1e-6)


def test_scheduled_decoder_inputs_preserves_answer_start():
    decoder = torch.tensor([[7, 8, 9, 10]])
    predictions = torch.tensor([[11, 12, 13, 14]])

    mixed = scheduled_decoder_inputs(decoder, predictions, probability=1.0)

    assert mixed.tolist() == [[7, 11, 12, 13]]
