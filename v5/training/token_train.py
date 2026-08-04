"""Train and save the small one-token attribute classifier."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import load_records
from .token_dataset import records_to_tensors
from .token_data import split_token_records
from .token_labels import ATOMIC_LABELS
from .token_model import OUTPUT_HEADS, TOKEN_MAX_LENGTH, TokenAttributeModel


def attribute_pos_weight(targets: dict[str, Tensor]) -> Tensor:
    positives = targets["attributes"].sum(dim=0)
    negatives = targets["attributes"].shape[0] - positives
    weights = torch.where(positives > 0, negatives / positives.clamp_min(1), torch.ones_like(positives))
    return weights.clamp(min=1, max=20)


def token_multitask_loss(
    logits: dict[str, Tensor], targets: dict[str, Tensor], *, pos_weight: Tensor | None = None,
) -> Tensor:
    attribute_loss = F.binary_cross_entropy_with_logits(
        logits["attribute_logits"], targets["attributes"], pos_weight=pos_weight,
    )
    active = targets["attributes"].bool()
    if active.any():
        delta_loss = F.cross_entropy(logits["delta_logits"][active], targets["deltas"][active])
    else:
        delta_loss = logits["delta_logits"].sum() * 0
    return attribute_loss + delta_loss


def train_token_model(
    records: list[dict[str, Any]], *, epochs: int = 300, learning_rate: float = 0.01, seed: int = 7,
) -> tuple[TokenAttributeModel, list[float]]:
    torch.manual_seed(seed)
    input_ids, targets = records_to_tensors(records)
    model = TokenAttributeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = attribute_pos_weight(targets)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = token_multitask_loss(model(input_ids), targets, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


def save_checkpoint(model: TokenAttributeModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {
            "label_count": model.label_count,
            "embedding_dim": model.embedding.embedding_dim,
            "hidden_dim": model.projection[0].out_features,
            "max_length": model.max_length,
        },
        "labels": ATOMIC_LABELS,
        "output_heads": OUTPUT_HEADS,
        "format_version": 1,
    }, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/training-token-attributes.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/token_ai.pt"))
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    records = load_records(args.data)
    splits = split_token_records(records)
    model, history = train_token_model(splits["train"], epochs=args.epochs)
    save_checkpoint(model, args.output)
    print({
        "train_records": len(splits["train"]),
        "dev_records": len(splits["dev"]),
        "test_records": len(splits["test"]),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "checkpoint": str(args.output),
    })


if __name__ == "__main__":
    main()
