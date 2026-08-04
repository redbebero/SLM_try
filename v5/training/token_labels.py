"""Stable atomic attribute labels for token model and ONNX consumers."""

from collections.abc import Mapping, Sequence

ATTRIBUTE_VALUES = {
    "ELEMENT": ("FIRE", "WATER", "AIR", "EARTH", "ICE", "LIGHT", "SHADOW", "LIFE", "POISON", "THUNDER", "VOID"),
    "FORM": ("ORB", "SPEAR", "SHIELD", "BEAM", "WALL", "WAVE", "ARROW", "CHAIN", "ZONE", "TRAP"),
    "TARGET": ("SELF", "ALLY", "ENEMY", "AREA", "GROUND", "OBJECT", "TOUCH", "PROJECTILE"),
    "INTENT": ("DAMAGE", "DEFEND", "CONTROL", "MOVE", "TRANSFORM", "RESTORE", "SENSE", "SUMMON", "DISPEL", "CREATE", "DESTROY", "TELEPORT", "ABSORB", "ENCHANT"),
    "MODIFIER": ("POWER_UP", "SPEED_UP", "RANGE_UP", "DURATION_UP", "PRECISION_UP", "AREA_UP", "PIERCE", "SEEK", "REPEAT", "REFLECT", "BURN", "FREEZE", "STUN", "SILENCE"),
    "SIZE": ("TINY", "SMALL", "MEDIUM", "LARGE", "HUGE", "ALL"),
    "DIRECTION": ("FORWARD", "BACKWARD", "UP", "DOWN", "LEFT", "RIGHT", "INWARD", "OUTWARD", "AROUND"),
    "QUANTITY": ("ONE", "FEW", "MANY", "ALL"),
    "CAST": ("CHARGE", "FOCUS", "RELEASE", "STOP", "CONTINUE", "NOW"),
}
ATOMIC_LABELS = tuple(f"{kind}:{value}" for kind, values in ATTRIBUTE_VALUES.items() for value in values)
DELTA_LABELS = (-2, -1, 0, 1, 2)


def _label(attribute: Mapping[str, object]) -> str:
    unsupported = set(attribute) - {"kind", "value", "delta"}
    if unsupported:
        raise ValueError(f"unsupported attribute field: {sorted(unsupported)[0]}")
    kind = attribute.get("kind")
    value = attribute.get("value")
    if kind not in ATTRIBUTE_VALUES or value not in ATTRIBUTE_VALUES[kind]:
        raise ValueError(f"invalid attribute label: {kind}:{value}")
    delta = attribute.get("delta")
    if not isinstance(delta, int) or delta not in DELTA_LABELS:
        raise ValueError(f"invalid attribute delta: {delta}")
    return f"{kind}:{value}"


def attribute_targets(attributes: Sequence[Mapping[str, object]]) -> dict[str, list[int]]:
    active = [0] * len(ATOMIC_LABELS)
    deltas = [DELTA_LABELS.index(0)] * len(ATOMIC_LABELS)
    for attribute in attributes:
        label = _label(attribute)
        index = ATOMIC_LABELS.index(label)
        active[index] = 1
        deltas[index] = DELTA_LABELS.index(attribute["delta"])
    return {"attributes": active, "deltas": deltas}


def targets_to_attributes(targets: Mapping[str, Sequence[int]]) -> list[dict[str, object]]:
    active = targets["attributes"]
    deltas = targets["deltas"]
    result: list[dict[str, object]] = []
    for index, flag in enumerate(active):
        if flag:
            kind, value = ATOMIC_LABELS[index].split(":", 1)
            result.append({"kind": kind, "value": value, "delta": DELTA_LABELS[deltas[index]]})
    return result
