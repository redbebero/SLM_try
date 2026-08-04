"""Decode token model logits into the strict word-classification contract."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .token_labels import DELTA_LABELS


def logits_to_token_classification(
    logits: Mapping[str, torch.Tensor],
    labels: Sequence[str],
    *,
    token: str,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    probabilities = torch.sigmoid(logits["attribute_logits"])
    delta_probabilities = torch.softmax(logits["delta_logits"], dim=-1)
    results: list[dict[str, Any]] = []
    for row in range(probabilities.shape[0]):
        attributes: list[dict[str, Any]] = []
        for index, label in enumerate(labels):
            confidence = float(probabilities[row, index])
            if confidence < threshold:
                continue
            kind, value = label.split(":", 1)
            delta_index = int(delta_probabilities[row, index].argmax())
            attributes.append({
                "kind": kind,
                "value": value,
                "delta": DELTA_LABELS[delta_index],
                "confidence": confidence,
            })
        results.append({"schema_version": 1, "token": token, "attributes": attributes, "unknown": not attributes})
    return results
