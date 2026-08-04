"""Build a balanced, disjoint public corpus for from-scratch Korean HRM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from build_scratch_corpus import load_jsonl, prompt_key, unique, write_dir


def take(path: Path, limit: int, rng: Random) -> list[dict]:
    rows = load_jsonl(path)
    rng.shuffle(rows)
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--empathetic-train", type=Path, required=True)
    parser.add_argument("--empathetic-valid", type=Path, required=True)
    parser.add_argument("--roleplay-train", type=Path, required=True)
    parser.add_argument("--roleplay-valid", type=Path, required=True)
    parser.add_argument("--dialogue-train", type=Path, required=True)
    parser.add_argument("--dialogue-valid", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, required=True)
    parser.add_argument("--general-valid", type=Path, required=True)
    args = parser.parse_args()
    rng = Random(args.seed)

    train_parts = [
        ("empathetic", take(args.empathetic_train, 9000, rng)),
        ("roleplay", take(args.roleplay_train, 4000, rng)),
        ("dialogue", take(args.dialogue_train, 6000, rng)),
        ("general", take(args.general_train, 8000, rng)),
    ]
    valid_parts = [
        ("empathetic", take(args.empathetic_valid, 1000, rng)),
        ("roleplay", take(args.roleplay_valid, 500, rng)),
        ("dialogue", take(args.dialogue_valid, 700, rng)),
        ("general", take(args.general_valid, 1000, rng)),
    ]
    train = unique([row for _, rows in train_parts for row in rows])
    valid = unique([row for _, rows in valid_parts for row in rows])
    valid_keys = {prompt_key(row) for row in valid}
    train = [row for row in train if prompt_key(row) not in valid_keys]
    rng.shuffle(train)
    rng.shuffle(valid)

    write_dir(args.output / "sft_train", train, True)
    write_dir(args.output / "sft_valid", valid, True)
    write_dir(args.output / "pretrain_train", train, False)
    write_dir(args.output / "pretrain_valid", valid, False)
    overlap = len({prompt_key(row) for row in train} & valid_keys)
    manifest = {
        "sources": [str(path) for path in (
            args.empathetic_train, args.empathetic_valid,
            args.roleplay_train, args.roleplay_valid,
            args.dialogue_train, args.dialogue_valid,
            args.general_train, args.general_valid,
        )],
        "train": len(train),
        "valid": len(valid),
        "seed": args.seed,
        "question_overlap": overlap,
        "mix": {name: len(rows) for name, rows in train_parts},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
