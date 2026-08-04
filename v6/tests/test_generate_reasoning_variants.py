import unittest
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path


class GenerateReasoningVariantsTest(unittest.TestCase):
    def test_generates_verified_records_disjoint_from_evaluation_ids(self):
        from scripts.generate_reasoning_variants import build_variants
        from scripts.build_reasoning_suite import build_suite

        rows = build_variants(120)
        test_rows, ood_rows = build_suite()
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({row["id"] for row in rows}), 120)
        self.assertTrue({row["id"] for row in rows}.isdisjoint({row["id"] for row in test_rows + ood_rows}))
        self.assertTrue(all(row["answer"] in row["solution"] for row in rows))

    def test_cli_writes_jsonl(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            subprocess.run([sys.executable, "scripts/generate_reasoning_variants.py", "--output", str(output), "--count", "6"], check=True)
            self.assertEqual(len(output.read_text().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
