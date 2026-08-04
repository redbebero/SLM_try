import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class BuildReasoningMixtureTest(unittest.TestCase):
    def test_builds_target_count_without_duplicate_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"id": "x", "question": "q", "solution": "a", "reasoning": "a", "answer": "a", "category": "arithmetic", "template_id": "x", "source": "s", "license": "MIT"}
            paths = []
            for name in ("owned", "korquad", "gsm", "instruction", "existing"):
                path = root / f"{name}.jsonl"
                path.write_text("".join(json.dumps({**row, "id": f"{name}-{i}"}) + "\n" for i in range(3)))
                paths.append(path)
            output = root / "train.jsonl"
            subprocess.run([sys.executable, "scripts/build_reasoning_mixture.py", "--output", str(output), "--target", "10", *sum((["--" + name, str(path)] for name, path in zip(("owned", "korquad", "gsm", "instruction", "existing"), paths)), [])], check=True)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row["id"] for row in rows}), 10)


if __name__ == "__main__":
    unittest.main()
