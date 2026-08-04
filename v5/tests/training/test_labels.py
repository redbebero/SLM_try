import pytest

from training.labels import proposal_to_targets, targets_to_proposal


def test_proposal_labels_have_stable_order_and_round_trip():
    proposal = {
        "schema_version": 1,
        "status": "PROPOSAL",
        "element": "FIRE",
        "form": "SPEAR",
        "target": "ENEMY",
        "power": 3,
        "speed": 2,
        "range": 1,
        "duration": 0,
        "confidence": 1.0,
    }

    targets = proposal_to_targets(proposal)
    restored = targets_to_proposal(targets, confidence=0.75)

    assert targets == {
        "status": 1,
        "element": 0,
        "form": 1,
        "target": 1,
        "power": 3,
        "speed": 2,
        "range": 1,
        "duration": 0,
    }
    assert restored == {**proposal, "confidence": 0.75}


def test_proposal_labels_reject_game_state_fields():
    with pytest.raises(ValueError, match="unsupported proposal field"):
        proposal_to_targets({"mana": 99})


def test_light_is_a_supported_proposal_element():
    proposal = {
        "schema_version": 1,
        "status": "PROPOSAL",
        "element": "LIGHT",
        "form": "BEAM",
        "target": "AREA",
        "power": 2,
        "speed": 2,
        "range": 2,
        "duration": 1,
        "confidence": 1.0,
    }

    assert targets_to_proposal(proposal_to_targets(proposal), confidence=1.0) == proposal
