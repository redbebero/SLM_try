from training.generate_dataset import expand_records


def _record(record_id: str, group: str, status: str) -> dict:
    unknown = status == "UNKNOWN"
    return {
        "id": record_id,
        "split_group": group,
        "input": {"incantation": "붉은 불꽃" if not unknown else "일반 문장", "language": "ko"},
        "target": {
            "schema_version": 1,
            "status": status,
            "element": "UNKNOWN" if unknown else "FIRE",
            "form": "UNKNOWN" if unknown else "ORB",
            "target": "UNKNOWN" if unknown else "ENEMY",
            "power": 0 if unknown else 1,
            "speed": 0 if unknown else 1,
            "range": 0 if unknown else 1,
            "duration": 0,
            "confidence": 1,
        },
        "provenance": {"type": "HARD_NEGATIVE" if unknown else "HUMAN_AUTHORED", "source_id": group, "verified": True},
    }


def test_expand_records_keeps_labels_and_groups_for_variants():
    records = expand_records([_record("positive", "group_positive", "PROPOSAL"), _record("negative", "group_negative", "UNKNOWN")], positive_variations=2, negative_variations=1)

    assert len(records) == 5
    assert len({record["id"] for record in records}) == 5
    assert {record["split_group"] for record in records if record["target"]["status"] == "PROPOSAL"} == {"group_positive"}
    assert all(record["target"]["status"] == "UNKNOWN" for record in records if record["split_group"] == "group_negative")
