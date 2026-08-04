"""Convert model logits into the JSON-compatible SpellProposal shape."""

from collections.abc import Mapping
from typing import Any

import torch

from .labels import targets_to_proposal
from .model import OUTPUT_HEADS


def logits_to_proposal(logits: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    batch_size = logits[OUTPUT_HEADS[0]].shape[0]
    predictions = {
        name: logits[name].argmax(dim=-1).tolist() for name in OUTPUT_HEADS
    }
    confidences = {
        name: torch.softmax(logits[name], dim=-1).max(dim=-1).values.tolist()
        for name in OUTPUT_HEADS
    }

    proposals: list[dict[str, Any]] = []
    for row in range(batch_size):
        confidence = min(confidences[name][row] for name in OUTPUT_HEADS)
        targets = {name: predictions[name][row] for name in OUTPUT_HEADS}
        proposal = targets_to_proposal(targets, confidence=float(confidence))
        if proposal["status"] == "UNKNOWN":
            proposal.update({
                "element": "UNKNOWN",
                "form": "UNKNOWN",
                "target": "UNKNOWN",
                "power": 0,
                "speed": 0,
                "range": 0,
                "duration": 0,
            })
        proposals.append(proposal)
    return proposals
