"""Prepare clean multi-turn Korean roleplay data without conversation leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd


def clean_text(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\*[^*]{1,180}\*", "", text).strip()
    text = text.strip('"“”')
    if not 2 <= len(text) <= 400:
        return ""
    if "http://" in text.lower() or "https://" in text.lower():
        return ""
    return text


def load_conversations(root: Path, general_limit: int, seed: int):
    paths = [
        root / "exa-data/train-00000-of-00001.parquet",
        root / "gf-persona-data/train-00000-of-00001.parquet",
        root / "youtube-couple-data/train-00000-of-00001.parquet",
    ]
    general = root / "general-roleplay-data/train-00000-of-00001.parquet"
    paths.append(general)
    conversations = []
    for path in paths:
        frame = pd.read_parquet(path)
        rows = list(frame.itertuples(index=False))
        if path == general:
            random.Random(seed).shuffle(rows)
            rows = rows[:general_limit]
        for row in rows:
            raw = getattr(row, "text")
            turns = []
            for turn in raw:
                role = str(turn.get("role", "")).lower()
                content = clean_text(turn.get("content"))
                if role not in ("user", "assistant") or not content:
                    continue
                if turns and turns[-1]["role"] == role:
                    turns[-1]["content"] += " " + content
                else:
                    turns.append({"role": role, "content": content})
            user_count = sum(turn["role"] == "user" for turn in turns)
            assistant_count = sum(turn["role"] == "assistant" for turn in turns)
            if user_count < 2 or assistant_count < 2:
                continue
            # Keep ordinary dialogue; discard obvious fiction/action residue.
            joined = " ".join(turn["content"] for turn in turns)
            if any(mark in joined for mark in ("###", "<|", "[INST]")):
                continue
            conversations.append(turns)
    return conversations


def expand(conversations):
    records = []
    for turns in conversations:
        history = []
        for turn in turns:
            if turn["role"] == "assistant" and history and history[-1]["role"] == "user":
                records.append({"messages": list(history), "answer": turn["content"]})
            history.append(turn)
            # Avoid very long prompts while retaining multiple previous turns.
            if len(history) > 8:
                history = history[-8:]
    result, seen = [], set()
    for record in records:
        key = hashlib.sha1(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_external/raw/korean-role-playing"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_external/processed/multiturn_roleplay"))
    parser.add_argument("--general-limit", type=int, default=5000)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    conversations = load_conversations(args.root, args.general_limit, args.seed)
    random.Random(args.seed).shuffle(conversations)
    valid_count = max(1, round(len(conversations) * args.valid_ratio))
    valid_conversations = conversations[:valid_count]
    train_conversations = conversations[valid_count:]
    train = expand(train_conversations)
    valid = expand(valid_conversations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in train), encoding="utf-8"
    )
    (args.output_dir / "valid.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in valid), encoding="utf-8"
    )
    manifest = {
        "conversations": len(conversations),
        "train_conversations": len(train_conversations),
        "valid_conversations": len(valid_conversations),
        "train_records": len(train),
        "valid_records": len(valid),
        "general_limit": args.general_limit,
        "license": "Apache-2.0",
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
