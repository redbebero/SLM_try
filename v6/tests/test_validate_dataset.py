import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class ValidateDatasetTest(unittest.TestCase):
    def test_validator_accepts_template_disjoint_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "owned.jsonl"
            output = root / "splits"
            subprocess.run([sys.executable, "scripts/build_owned_dataset.py", "--output", str(source)], check=True)
            subprocess.run([sys.executable, "scripts/split_dataset.py", "--input", str(source), "--output-dir", str(output)], check=True)
            result = subprocess.run([sys.executable, "scripts/validate_dataset.py", "--data-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
