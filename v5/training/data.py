"""Dataset loading and leakage-safe group splitting."""

import json
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"record at line {line_number} must be an object")
        records.append(value)
    if not records:
        raise ValueError("dataset must contain at least one record")
    return records


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split by complete-incantation group using stable round-robin buckets."""
    groups = sorted({str(record["split_group"]) for record in records})
    group_to_split = {
        group: ("train" if index % 10 < 8 else "dev" if index % 10 == 8 else "test")
        for index, group in enumerate(groups)
    }
    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        splits[group_to_split[str(record["split_group"])]].append(record)
    return splits
