import torch

from pretrain_subword_lm import chunk_ids


def test_chunk_ids_preserves_order_and_limits_length():
    ids = torch.arange(10)
    chunks = chunk_ids(ids, max_len=4)
    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8], [8, 9]]
