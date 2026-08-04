"""Download public Korean datasets and preserve source metadata locally."""

import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "korean_quality_cleaned": {
        "hub_id": "MyeongHo0621/korean-quality-cleaned",
        "license": "source dataset card; inspect component licenses before training",
    },
    "aihub_education_sample": {
        "hub_id": "neuralfoundry-coder/aihub-korean-education-instruct-sample",
        "license": "CC BY-NC-SA 4.0; AI Hub terms apply",
    },
}


def main():
    root = ROOT / "datasets" / "external"
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "datasets": []}
    for name, info in TARGETS.items():
        dataset = load_dataset(info["hub_id"])
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        splits = {}
        for split, rows in dataset.items():
            path = target / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            splits[split] = {"path": str(path.relative_to(ROOT)), "rows": len(rows)}
        manifest["datasets"].append({
            "name": name,
            "hub_id": info["hub_id"],
            "license": info["license"],
            "splits": splits,
        })
    (root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
