import sys

sys.path.insert(0, "scripts")

from conditional_selector import ConditionalSelector


def test_selector_returns_exact_downloaded_pair_before_fuzzy_hits():
    rows = [
        {"question": "오늘 날씨 어때?", "answer": "맑아요.", "pair_hash": "one"},
        {"question": "오늘 기분 어때?", "answer": "좋아요.", "pair_hash": "two"},
    ]
    selector = ConditionalSelector(rows)

    result = selector.select("오늘 날씨 어때?")

    assert result["answer"] == "맑아요."
    assert result["method"] == "exact"
    assert result["score"] == 1.0


def test_selector_can_exclude_current_pair_from_train_retrieval():
    rows = [
        {"question": "같은 질문", "answer": "첫 답", "pair_hash": "one"},
        {"question": "같은 질문", "answer": "둘째 답", "pair_hash": "two"},
    ]
    selector = ConditionalSelector(rows)

    result = selector.select("같은 질문", exclude_pair_hash="one")

    assert result["pair_hash"] == "two"


def test_selector_rejects_below_confidence_threshold():
    selector = ConditionalSelector([
        {"question": "완전히 다른 질문", "answer": "답", "pair_hash": "one"},
    ])

    result = selector.select("관련 없는 입력", min_score=0.9)

    assert result is None
