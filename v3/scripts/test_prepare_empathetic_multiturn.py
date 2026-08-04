import unittest

from prepare_empathetic_multiturn import parse_instruction


class EmpatheticParsingTest(unittest.TestCase):
    def test_extracts_alternating_context_and_final_answer(self):
        row = parse_instruction(
            "질문: 요즘 힘들어요.\n답변: 많이 힘드시겠어요.\n"
            "질문: 잠도 잘 못 자요.",
            "걱정이 많으셨군요. 오늘은 조금 쉬어도 괜찮아요.",
        )
        self.assertEqual(row["messages"][0]["role"], "user")
        self.assertEqual(row["messages"][1]["role"], "assistant")
        self.assertEqual(row["messages"][-1]["content"], "잠도 잘 못 자요.")
        self.assertEqual(row["answer"], "걱정이 많으셨군요. 오늘은 조금 쉬어도 괜찮아요.")

    def test_single_turn_is_supported(self):
        row = parse_instruction("새 취미를 찾고 있어요.", "어떤 활동을 좋아하세요?")
        self.assertEqual(row["messages"], [{"role": "user", "content": "새 취미를 찾고 있어요."}])


if __name__ == "__main__":
    unittest.main()
