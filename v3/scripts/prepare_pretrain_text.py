"""Extract short Korean conversational lines for HRM language pretraining."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from random import Random


def clean(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not 2 <= len(text) <= 180 or "http" in text.lower():
        return ""
    korean = len(re.findall(r"[가-힣]", text))
    if korean < 2:
        return ""
    return text


def blocks(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").split("\n\n") if x.strip()]


def make_lines(sft_paths: list[Path], culture_parquet: Path | None, limit: int, seed: int) -> list[str]:
    lines = []
    for path in sft_paths:
        for block in blocks(path):
            for line in block.splitlines():
                if line.startswith(("Q: ", "A: ")):
                    value = clean(line[3:])
                    if value:
                        lines.append(value)
    if culture_parquet:
        import pandas as pd
        frame = pd.read_parquet(culture_parquet)
        for _, row in frame.iterrows():
            for key in ("question", "answer"):
                value = clean(row.get(key, ""))
                if value:
                    lines.append(value)
    unique = list(dict.fromkeys(lines))
    Random(seed).shuffle(unique)
    unique = unique[:limit]
    # The non-SFT dataset makes windows within each line. Pack short turns so
    # the pretraining loader has enough context for 256-token windows.
    return [" ".join(unique[index:index + 20]) for index in range(0, len(unique), 20)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", nargs="+", type=Path, default=[Path("data_external/processed/external_dialogue_sft.txt")])
    parser.add_argument("--culture", type=Path, default=Path("data_external/raw/KoCulture-Dialogues/data/train-00000-of-00001.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_external/pretrain"))
    parser.add_argument("--limit", type=int, default=15000)
    args = parser.parse_args()
    lines = make_lines(args.sft, args.culture if args.culture.exists() else None, args.limit, 42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "korean_dialogue.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"lines": len(lines), "seed": 42, "max_chars": 180, "source_culture": args.culture.exists()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
