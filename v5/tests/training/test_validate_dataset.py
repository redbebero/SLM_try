import json

from training.validate_dataset import validate_dataset


def _record() -> dict:
    return {
        "id": "spell_1",
        "split_group": "group_1",
        "input": {"incantation": "붉은 불꽃", "language": "ko"},
        "target": {
            "schema_version": 1,
            "status": "PROPOSAL",
            "element": "FIRE",
            "form": "ORB",
            "target": "ENEMY",
            "power": 1,
            "speed": 1,
            "range": 1,
            "duration": 0,
            "confidence": 1,
        },
        "provenance": {"type": "HUMAN_AUTHORED", "source_id": "group_1", "verified": True},
    }


def test_validate_dataset_accepts_valid_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    assert validate_dataset(path) == []


def test_validate_dataset_reports_line_and_schema_error(tmp_path):
    path = tmp_path / "records.jsonl"
    invalid = _record()
    invalid["target"]["mana"] = 99
    path.write_text(json.dumps(invalid, ensure_ascii=False) + "\n", encoding="utf-8")

    errors = validate_dataset(path)

    assert errors
    assert errors[0].startswith("line 1:")
    assert "additionalProperties" in errors[0]
