import unittest

import torch

from scripts.train_lora import format_record


class TrainLoraTest(unittest.TestCase):
    def test_format_contains_question_solution_and_answer(self):
        text = format_record({"question": "2 더하기 2는?", "solution": "계산하면 4입니다.", "answer": "4"})
        self.assertIn("2 더하기 2는?", text)
        self.assertIn("계산하면 4입니다.", text)
        self.assertIn("정답: 4", text)

    def test_collate_pads_variable_length_examples_and_masks_padding(self):
        from scripts.train_lora import collate_examples

        batch = collate_examples([{"input_ids": [1, 2], "attention_mask": [1, 1]}, {"input_ids": [3], "attention_mask": [1]}], pad_token_id=0)
        self.assertTrue(torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 0]])))
        self.assertTrue(torch.equal(batch["labels"], torch.tensor([[1, 2], [3, -100]])))


if __name__ == "__main__":
    unittest.main()
