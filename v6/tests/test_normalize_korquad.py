import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class NormalizeKorquadTest(unittest.TestCase):
    def test_normalizer_keeps_question_context_and_answer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            output = root / "normalized.jsonl"
            source.write_text(json.dumps({
                "id": "1",
                "title": "문서",
                "context": "<p>서울은 <b>한국</b>의 수도다.</p>",
                "question": "한국의 수도는?",
                "answer": {"text": "서울"},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            subprocess.run([sys.executable, "scripts/normalize_korquad.py", "--input", str(source), "--output", str(output)], check=True)
            item = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(item["answer"], "서울")
        self.assertIn("서울은 한국의 수도다", item["question"])
        self.assertEqual(item["category"], "document_qa")


if __name__ == "__main__":
    unittest.main()
