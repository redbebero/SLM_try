import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class NormalizeReasoningDataTest(unittest.TestCase):
    def test_normalizer_adds_source_metadata_and_preserves_answers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"id": "x", "question": "1+1?", "solution": "1+1=2", "answer": "2", "category": "arithmetic", "template_id": "t"}) + "\n")
            output = root / "output.jsonl"
            subprocess.run([sys.executable, "scripts/normalize_reasoning_data.py", "--input", str(source), "--output", str(output), "--source", "test", "--license", "MIT"], check=True)
            row = json.loads(output.read_text())
        self.assertEqual(row["answer"], "2")
        self.assertEqual(row["source"], "test")
        self.assertEqual(row["license"], "MIT")


if __name__ == "__main__":
    unittest.main()
