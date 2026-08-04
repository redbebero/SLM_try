import torch

from seq2seq_lm import CompactSeq2SeqLM


def test_seq2seq_is_small_and_returns_decoder_logits():
    model = CompactSeq2SeqLM(vocab_size=2048, emb_dim=16, hidden_dim=64,
                             layers=1, heads=4, max_src_len=128, max_tgt_len=128)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
    source = torch.randint(0, 2048, (2, 12))
    target = torch.randint(0, 2048, (2, 9))
    logits = model(source, target)
    assert logits.shape == (2, 9, 2048)


def test_seq2seq_decoder_is_causal():
    torch.manual_seed(4)
    model = CompactSeq2SeqLM(vocab_size=64, emb_dim=8, hidden_dim=32,
                             layers=1, heads=4, max_src_len=32, max_tgt_len=32)
    model.eval()
    source = torch.randint(0, 64, (1, 6))
    target = torch.randint(0, 64, (1, 5))
    changed = target.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 64
    with torch.no_grad():
        first = model(source, target)[:, :-1]
        second = model(source, changed)[:, :-1]
    assert torch.allclose(first, second, atol=1e-5)
