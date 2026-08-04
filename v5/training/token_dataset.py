"""Tensor conversion for one-token multi-label records."""

from collections.abc import Sequence
from typing import Any

import torch

from .encoding import encode_incantation
from .token_labels import attribute_targets
from .token_model import TOKEN_MAX_LENGTH


def records_to_tensors(records: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not records:
        raise ValueError("records must not be empty")
    input_ids = torch.tensor(
        [encode_incantation(record["input"]["token"], TOKEN_MAX_LENGTH) for record in records],
        dtype=torch.long,
    )
    encoded = [attribute_targets(record["target"]) for record in records]
    targets = {
        "attributes": torch.tensor([row["attributes"] for row in encoded], dtype=torch.float32),
        "deltas": torch.tensor([row["deltas"] for row in encoded], dtype=torch.long),
    }
    return input_ids, targets
