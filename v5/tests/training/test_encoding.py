import pytest

from training.encoding import PAD_ID, MAX_LENGTH, encode_incantation


def test_encode_incantation_uses_utf8_bytes_and_deterministic_padding():
    encoded = encode_incantation("불")

    assert len(encoded) == MAX_LENGTH
    assert encoded[:3] == [byte + 1 for byte in "불".encode("utf-8")]
    assert encoded[3:] == [PAD_ID] * (MAX_LENGTH - 3)


def test_encode_incantation_truncates_long_input_deterministically():
    text = "불" * 100

    assert encode_incantation(text) == encode_incantation(text)
    assert len(encode_incantation(text)) == MAX_LENGTH
    assert PAD_ID not in encode_incantation(text)


def test_encode_incantation_rejects_empty_input():
    with pytest.raises(ValueError, match="incantation must not be empty"):
        encode_incantation("   ")
