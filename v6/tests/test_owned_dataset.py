import json
import subprocess
import sys
import unittest
from pathlib import Path


class OwnedDatasetTest(unittest.TestCase):
    def test_builder_creates_300_valid_records(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            output = Path(directory) / "owned.jsonl"
            subprocess.run(
                [sys.executable, "scripts/build_owned_dataset.py", "--output", str(output)],
                check=True,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 300)
        self.assertEqual(
            {row["category"] for row in rows},
            {"arithmetic", "multi_step", "comparison", "document_qa", "state_change"},
        )
        self.assertEqual(len({row["template_id"] for row in rows}), 50)
        self.assertTrue(all(row["question"] and row["answer"] and row["solution"] for row in rows))


if __name__ == "__main__":
    unittest.main()
