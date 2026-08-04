import sys

import torch

sys.path.insert(0, "scripts")

from syllable_core import SyllableTokenizer
from full_syllable_lm import FullSyllableCausalLM
from train_full_syllable import encode_record
from evaluate_full_syllable import generation_context


def test_full_syllable_decoder_is_causal_and_small():
    tokenizer = SyllableTokenizer()
    model = FullSyllableCausalLM(
        vocab_size=tokenizer.get_vocab_size(),
        jamo_vocab_sizes=tokenizer.jamo.get_vocab_sizes(),
        emb_dim=16,
        hidden_dim=64,
        layers=1,
        heads=4,
        max_len=128,
    ).eval()
    ids = tokenizer.encode("Q: 안녕?|").unsqueeze(0)
    jamo = tokenizer.jamo_ids(ids[0]).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids, jamo)
        changed = ids.clone()
        changed[:, -1] = tokenizer.stoi["가"]
        changed_jamo = tokenizer.jamo_ids(changed[0]).unsqueeze(0)
        changed_logits = model(changed, changed_jamo)
    assert logits.shape == (1, ids.size(1), tokenizer.get_vocab_size())
    assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6)
    assert sum(parameter.numel() for parameter in model.parameters()) < 4_000_000


def test_full_syllable_decoder_supports_standalone_korean_jamo():
    tokenizer = SyllableTokenizer()
    text = "ㅋㅋ ㅠㅠ ㄹㅇ"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_full_syllable_sft_target_starts_after_delimiter_and_ends_eos():
    tokenizer = SyllableTokenizer()
    x, y, mask = encode_record(
        {"question": "안녕", "answer": "반가워"}, tokenizer
    )
    delimiter = tokenizer.encode("Q: 안녕|")
    assert mask[: len(delimiter) - 1].sum().item() == 0
    assert tokenizer.decode(y[len(delimiter) - 1:]).startswith("반가워")
    assert tokenizer.decode(y[-1:]) == "\n"


def test_generation_context_keeps_latest_tokens_within_model_window():
    ids = torch.arange(300)

    context = generation_context(ids, max_len=128)

    assert context.shape == (128,)
    assert context.tolist() == list(range(172, 300))
