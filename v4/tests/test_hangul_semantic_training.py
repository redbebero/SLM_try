import sys

sys.path.insert(0, "scripts")

from train_hangul_semantic import same_category_batches


def test_hard_negative_batches_keep_category_together():
    rows = [
        {"category": "a"}, {"category": "a"},
        {"category": "b"}, {"category": "b"},
    ]
    batches = same_category_batches(rows, batch_size=2, seed=3)

    assert sorted(sorted(batch) for batch in batches) == [[0, 1], [2, 3]]
