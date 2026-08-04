from training.token_labels import ATOMIC_LABELS, attribute_targets, targets_to_attributes


def test_atomic_labels_have_stable_order_and_round_trip_multiple_attributes():
    attributes = [
        {"kind": "ELEMENT", "value": "FIRE", "delta": 1},
        {"kind": "FORM", "value": "ORB", "delta": 0},
    ]

    targets = attribute_targets(attributes)

    assert targets["attributes"][ATOMIC_LABELS.index("ELEMENT:FIRE")] == 1
    assert targets["attributes"][ATOMIC_LABELS.index("FORM:ORB")] == 1
    assert targets_to_attributes(targets) == attributes


def test_game_state_fields_are_not_supported():
    try:
        attribute_targets([{"kind": "ELEMENT", "value": "FIRE", "delta": 1, "mana": 99}])
    except ValueError as error:
        assert "unsupported attribute field" in str(error)
    else:
        raise AssertionError("game-state field must be rejected")
