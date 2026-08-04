from training.validate_token_dataset import validate_token_dataset


def test_validate_token_dataset_rejects_game_state_field(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"id":"x","split_group":"x","input":{"token":"불","language":"ko"},"target":[],"provenance":{"type":"HUMAN_AUTHORED","source_id":"x","verified":true},"mana":3}\n',
        encoding="utf-8",
    )

    errors = validate_token_dataset(path)

    assert errors
    assert any("additionalProperties" in error for error in errors)


def test_validate_token_dataset_rejects_unknown_attribute_value(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"id":"x","split_group":"x","input":{"token":"불","language":"ko"},"target":[{"kind":"ELEMENT","value":"NOT_A_REAL_ELEMENT","delta":1}],"provenance":{"type":"HUMAN_AUTHORED","source_id":"x","verified":true}}\n',
        encoding="utf-8",
    )

    errors = validate_token_dataset(path)

    assert errors
    assert any("enum" in error for error in errors)
