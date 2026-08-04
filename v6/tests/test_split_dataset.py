import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class SplitDatasetTest(unittest.TestCase):
    def test_split_is_template_disjoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "owned.jsonl"
            output = root / "splits"
            subprocess.run([sys.executable, "scripts/build_owned_dataset.py", "--output", str(source)], check=True)
            subprocess.run([sys.executable, "scripts/split_dataset.py", "--input", str(source), "--output-dir", str(output)], check=True)
            splits = {
                name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
                for name in ("train", "dev", "test", "ood")
            }
        self.assertEqual(sum(map(len, splits.values())), 300)
        seen = set()
        for rows in splits.values():
            templates = {row["template_id"] for row in rows}
            self.assertTrue(seen.isdisjoint(templates))
            seen.update(templates)


if __name__ == "__main__":
    unittest.main()
