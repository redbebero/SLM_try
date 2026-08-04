import json

from training.data import load_records, split_records


def _record(group: str, index: int) -> dict:
    return {
        "id": f"spell_{index}",
        "split_group": group,
        "input": {"incantation": f"주문 {index}", "language": "ko"},
        "target": {
            "schema_version": 1,
            "status": "UNKNOWN",
            "element": "UNKNOWN",
            "form": "UNKNOWN",
            "target": "UNKNOWN",
            "power": 0,
            "speed": 0,
            "range": 0,
            "duration": 0,
            "confidence": 1,
        },
        "provenance": {"type": "HARD_NEGATIVE", "source_id": group, "verified": True},
    }


def test_load_records_reads_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(_record(f"group_{i}", i), ensure_ascii=False) for i in range(3)))

    records = load_records(path)

    assert len(records) == 3
    assert records[0]["input"]["language"] == "ko"


def test_split_records_keeps_split_groups_together():
    records = [_record(f"group_{i}", i) for i in range(30)]

    splits = split_records(records)
    groups = {
        split: {record["split_group"] for record in split_records_list}
        for split, split_records_list in splits.items()
    }

    assert groups["train"].isdisjoint(groups["dev"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["dev"].isdisjoint(groups["test"])
    assert sum(len(items) for items in groups.values()) == 30
