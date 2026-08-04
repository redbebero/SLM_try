import json
import sys

sys.path.insert(0, "scripts")

from build_category_sft import keep_row, write_split


def test_category_split_writes_original_question_and_answer(tmp_path):
    rows = [{"question": "질문", "answer": "답변", "category": "dialogue"}]

    write_split(tmp_path / "train", rows)

    saved = json.loads((tmp_path / "train" / "records.jsonl").read_text(encoding="utf-8"))
    text = (tmp_path / "train" / "verified.txt").read_text(encoding="utf-8")
    assert saved == rows[0]
    assert "Q: 질문\nA: 답변" in text


def test_category_filter_keeps_only_short_downloaded_rows():
    row = {"question": "짧은 질문", "answer": "짧은 답"}
    long_row = {"question": "짧은 질문", "answer": "긴" * 81}

    assert keep_row(row, max_question=80, max_answer=80)
    assert not keep_row(long_row, max_question=80, max_answer=80)
