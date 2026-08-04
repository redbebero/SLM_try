import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASETS = [
    {
        "name": "openai/gsm8k",
        "config": "main",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
        "purpose": "English multi-step math translated and verifier-checked into Korean",
        "split": "train",
    },
    {
        "name": "LGCNS/KorQuAD_2.0",
        "config": "default",
        "revision": "383f6a3d4efd5f238b4df7181d0af182f0ea8ff",
        "license": "CC BY-ND 2.0 KR",
        "purpose": "Korean document, table, and list QA",
        "split": "train",
    },
]


def download(output_dir: Path, reuse_korquad: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in DATASETS:
        target = output_dir / item["name"].replace("/", "__")
        if item["name"] == "LGCNS/KorQuAD_2.0" and reuse_korquad:
            rows = sum(1 for _ in reuse_korquad.open(encoding="utf-8"))
            manifest.append({**item, "source_path": str(reuse_korquad), "rows": rows, "reused": True})
            continue
        dataset = load_dataset(item["name"], item["config"], split=item["split"], revision=item["revision"])
        target.mkdir(parents=True, exist_ok=True)
        dataset.to_json(target / f"{item['split']}.jsonl", force_ascii=False)
        manifest.append({**item, "source_path": str(target / f"{item['split']}.jsonl"), "rows": len(dataset), "reused": False})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-korquad", type=Path)
    args = parser.parse_args()
    for item in download(args.output_dir, args.reuse_korquad):
        print(item["name"], item["rows"], item["license"])


if __name__ == "__main__":
    main()
