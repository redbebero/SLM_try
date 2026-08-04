import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_safe_conversation import clean_record, split_records


class PrepareSafeConversationTests(unittest.TestCase):
  def test_clean_record_rejects_ai_disclaimer_and_long_answer(self):
    self.assertIsNone(clean_record({
        "instruction": "어제 뭐 했어?",
        "input": "",
        "output": "저는 인공지능이라서 어제 아무것도 하지 않았습니다.",
    }))
    self.assertIsNone(clean_record({
        "instruction": "짧게 답해줘",
        "input": "",
        "output": "가나다 " * 200,
    }))

  def test_clean_record_keeps_short_natural_answer(self):
    item = clean_record({
        "instruction": "오늘 기분이 어때?",
        "input": "",
        "output": "괜찮아. 네 이야기도 듣고 싶어.",
    })
    self.assertEqual(item, {
        "messages": [{"role": "user", "content": "오늘 기분이 어때?"}],
        "answer": "괜찮아. 네 이야기도 듣고 싶어.",
    })

  def test_split_records_is_deterministic_and_disjoint(self):
    records = [
        {"messages": [{"role": "user", "content": f"질문 {i}"}], "answer": f"답변 {i}"}
        for i in range(20)
    ]
    train_a, valid_a = split_records(records, valid_ratio=0.2)
    train_b, valid_b = split_records(records, valid_ratio=0.2)
    self.assertEqual(train_a, train_b)
    self.assertEqual(valid_a, valid_b)
    self.assertFalse(set(map(str, train_a)) & set(map(str, valid_a)))
    self.assertEqual(len(train_a) + len(valid_a), len(records))


if __name__ == "__main__":
    unittest.main()
