"""Stable label order shared by training and ONNX consumers."""

from collections.abc import Mapping

STATUS_LABELS = ("UNKNOWN", "PROPOSAL")
ELEMENT_LABELS = ("FIRE", "WATER", "AIR", "EARTH", "ICE", "LIGHT", "SHADOW", "UNKNOWN")
FORM_LABELS = ("ORB", "SPEAR", "SHIELD", "BEAM", "UNKNOWN")
TARGET_LABELS = ("SELF", "ENEMY", "AREA", "UNKNOWN")

LABELS_BY_FIELD = {
    "status": STATUS_LABELS,
    "element": ELEMENT_LABELS,
    "form": FORM_LABELS,
    "target": TARGET_LABELS,
}
_NUMERIC_FIELDS = ("power", "speed", "range", "duration")
_PROPOSAL_FIELDS = {
    "schema_version", "status", "element", "form", "target",
    "power", "speed", "range", "duration", "confidence",
}


def proposal_to_targets(proposal: Mapping[str, object]) -> dict[str, int]:
    unsupported = set(proposal) - _PROPOSAL_FIELDS
    if unsupported:
        raise ValueError(f"unsupported proposal field: {sorted(unsupported)[0]}")

    targets: dict[str, int] = {}
    for name, labels in LABELS_BY_FIELD.items():
        value = proposal.get(name)
        if value not in labels:
            raise ValueError(f"invalid {name} label: {value}")
        targets[name] = labels.index(value)

    for name in _NUMERIC_FIELDS:
        value = proposal.get(name)
        if not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"invalid {name} value: {value}")
        targets[name] = value

    return targets


def targets_to_proposal(targets: Mapping[str, int], confidence: float) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "status": STATUS_LABELS[targets["status"]],
        "element": ELEMENT_LABELS[targets["element"]],
        "form": FORM_LABELS[targets["form"]],
        "target": TARGET_LABELS[targets["target"]],
        "power": targets["power"],
        "speed": targets["speed"],
        "range": targets["range"],
        "duration": targets["duration"],
        "confidence": confidence,
    }
    return result
