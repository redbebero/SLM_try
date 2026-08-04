"""Tokenizer-free UTF-8 input encoding for the spell proposal model."""

PAD_ID = 0
MAX_LENGTH = 96


def encode_incantation(incantation: str, max_length: int = MAX_LENGTH) -> list[int]:
    """Encode Korean text as shifted UTF-8 bytes with zero padding."""
    if not incantation.strip():
        raise ValueError("incantation must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    byte_ids = [byte + 1 for byte in incantation.encode("utf-8")[:max_length]]
    return byte_ids + [PAD_ID] * (max_length - len(byte_ids))
