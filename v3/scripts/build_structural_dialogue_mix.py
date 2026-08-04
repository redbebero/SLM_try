"""Build a leakage-free dialogue mix for the structural student."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def blocks(path: Path) -> list[str]:
    return [item.strip() for item in path.read_text(encoding="utf-8").split("\n\n") if item.strip()]


def question(block: str) -> str:
    match = re.search(r"(?m)^Q:\s*(.+)$", block)
    return match.group(1).strip() if match else ""


def key(block: str) -> str:
    return hashlib.sha1(question(block).encode("utf-8")).hexdigest()


def dedupe(items: list[str]) -> list[str]:
    result, seen = [], set()
    for item in items:
        digest = key(item)
        if not question(item) or digest in seen:
            continue
        seen.add(digest)
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=Path("train_data_external_mix/train.txt"))
    parser.add_argument("--external-valid", type=Path, default=Path("train_data_external_mix_valid/valid.txt"))
    parser.add_argument("--koculture", type=Path, default=Path("data_external/processed/koculture_sft.txt"))
    parser.add_argument("--distill", type=Path, default=None)
    parser.add_argument("--output-train", type=Path, default=Path("train_data_structural_mix"))
    parser.add_argument("--output-valid", type=Path, default=Path("train_data_structural_mix_valid"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    external = dedupe(blocks(args.external))
    external_valid = dedupe(blocks(args.external_valid))
    culture = dedupe(blocks(args.koculture))
    distill = dedupe(blocks(args.distill)) if args.distill else []
    random.Random(args.seed).shuffle(culture)
    valid_count = max(1, round(len(culture) * args.valid_ratio))
    culture_valid, culture_train = culture[:valid_count], culture[valid_count:]

    # Remove question overlap across every split, including the old external validation set.
    valid_keys = {key(item) for item in external_valid + culture_valid}
    train = [item for item in external + culture_train + distill if key(item) not in valid_keys]
    valid = external_valid + culture_valid
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed).shuffle(valid)

    args.output_train.mkdir(parents=True, exist_ok=True)
    args.output_valid.mkdir(parents=True, exist_ok=True)
    (args.output_train / "train.txt").write_text("\n\n".join(train) + "\n", encoding="utf-8")
    (args.output_valid / "valid.txt").write_text("\n\n".join(valid) + "\n", encoding="utf-8")
    manifest = {
        "external_train": len(external),
        "koculture_train": len(culture_train),
        "external_valid": len(external_valid),
        "koculture_valid": len(culture_valid),
        "distill": len(distill),
        "train": len(train),
        "valid": len(valid),
        "question_overlap": len({key(item) for item in train} & {key(item) for item in valid}),
        "seed": args.seed,
    }
    (args.output_train / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
