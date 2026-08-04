import unittest

from scripts.evaluate_ko_problems import configure_tokenizer, normalize


class EvaluatorTest(unittest.TestCase):
    def test_configure_tokenizer_left_pads_decoder_only_inputs(self):
        class Tokenizer:
            padding_side = "right"

        self.assertEqual(configure_tokenizer(Tokenizer()).padding_side, "left")

    def test_normalize_removes_number_punctuation_and_spaces(self):
        self.assertEqual(normalize(" 정답은 1,200개입니다. "), "정답은1200개입니다")


if __name__ == "__main__":
    unittest.main()
