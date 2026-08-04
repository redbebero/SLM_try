"""Build a conservative replay mix for external Korean dialogue SFT."""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random


def read_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def build_mix(external_path: Path, replay_path: Path, output_dir: Path, external_limit: int = 5000, seed: int = 42) -> dict:
    external = read_blocks(external_path)
    replay = read_blocks(replay_path)
    Random(seed).shuffle(external)
    external = external[:external_limit]
    Random(seed).shuffle(replay)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_blocks = replay + external
    Random(seed).shuffle(train_blocks)
    (output_dir / "train.txt").write_text("\n\n".join(train_blocks) + "\n", encoding="utf-8")
    return {"replay": len(replay), "external": len(external), "train": len(train_blocks), "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=Path("data_external/processed/external_dialogue_sft.txt"))
    parser.add_argument("--external-valid", type=Path, default=Path("data_external/processed/external_dialogue_valid.txt"))
    parser.add_argument("--replay", type=Path, default=Path("train_data_hrm_dialogue_pure_v2/clean_dialogue_sft.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("train_data_external_mix"))
    parser.add_argument("--external-limit", type=int, default=5000)
    args = parser.parse_args()
    manifest = build_mix(args.external, args.replay, args.output_dir, args.external_limit)
    valid_dir = args.output_dir.with_name(args.output_dir.name + "_valid")
    valid_dir.mkdir(parents=True, exist_ok=True)
    (valid_dir / "valid.txt").write_text(args.external_valid.read_text(encoding="utf-8"), encoding="utf-8")
    print(manifest | {"valid": len(read_blocks(args.external_valid)), "valid_dir": str(valid_dir)})


if __name__ == "__main__":
    main()
