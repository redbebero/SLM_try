"""Load data-driven one-token lexicon into training records."""

import json
from pathlib import Path
from typing import Any

from .token_labels import attribute_targets


def lexicon_to_records(lexicon: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in lexicon.get("entries", []):
        entry_id = entry["id"]
        attributes = entry["attributes"]
        attribute_targets(attributes)
        for index, token in enumerate(entry["surfaces"]):
            if not isinstance(token, str) or not token.strip() or len(token.split()) != 1:
                raise ValueError(f"surface must be one non-empty token: {token!r}")
            records.append({
                "id": f"token_{entry_id}_{index:04d}",
                "split_group": entry_id,
                "input": {"token": token, "language": "ko"},
                "target": attributes,
                "provenance": entry["provenance"],
            })
    if not records:
        raise ValueError("lexicon must contain at least one surface")
    return records


def load_lexicon(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_token_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Hold out surface forms within each semantic entry, not whole labels."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["split_group"]), []).append(record)

    splits = {"train": [], "dev": [], "test": []}
    for group in sorted(grouped):
        group_records = grouped[group]
        total = len(group_records)
        train_end = max(1, int(total * 0.8))
        dev_end = max(train_end, int(total * 0.9))
        splits["train"].extend(group_records[:train_end])
        splits["dev"].extend(group_records[train_end:dev_end])
        splits["test"].extend(group_records[dev_end:])
    return splits


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
