"""Tensor conversion for validated training records."""

from collections.abc import Sequence
from typing import Any

import torch

from .encoding import encode_incantation
from .labels import proposal_to_targets
from .model import OUTPUT_HEADS


def records_to_tensors(records: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not records:
        raise ValueError("records must not be empty")

    input_ids = torch.tensor(
        [encode_incantation(record["input"]["incantation"]) for record in records],
        dtype=torch.long,
    )
    target_rows = [proposal_to_targets(record["target"]) for record in records]
    targets = {
        name: torch.tensor([row[name] for row in target_rows], dtype=torch.long)
        for name in OUTPUT_HEADS
    }
    return input_ids, targets
