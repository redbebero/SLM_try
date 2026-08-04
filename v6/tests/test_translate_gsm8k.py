import unittest

from scripts.translate_gsm8k import parse_gsm8k_answer


class TranslateGsm8kTest(unittest.TestCase):
    def test_extracts_final_answer(self):
        self.assertEqual(parse_gsm8k_answer("계산 과정\n#### 42"), "42")


if __name__ == "__main__":
    unittest.main()
