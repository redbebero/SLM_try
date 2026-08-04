import unittest


class VerifyTranslatedReasoningTest(unittest.TestCase):
    def test_accepts_answer_in_solution_and_rejects_missing_answer(self):
        from scripts.verify_translated_reasoning import verify_row

        good = {"question": "한 개", "solution": "1 + 1 = 2입니다.", "answer": "2"}
        bad = {"question": "한 개", "solution": "계산할 수 없습니다.", "answer": "2"}
        self.assertIsNone(verify_row(good))
        self.assertIn("answer", verify_row(bad))


if __name__ == "__main__":
    unittest.main()
