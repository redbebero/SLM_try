"""Train and save the small Korean spell proposal model."""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import load_records, split_records
from .dataset import records_to_tensors
from .model import OUTPUT_HEADS, SpellProposalModel


def multitask_loss(logits: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    return sum(F.cross_entropy(logits[name], targets[name]) for name in OUTPUT_HEADS)


def train_model(
    records: list[dict[str, Any]],
    *,
    epochs: int = 300,
    learning_rate: float = 0.01,
    seed: int = 7,
) -> tuple[SpellProposalModel, list[float]]:
    torch.manual_seed(seed)
    input_ids, targets = records_to_tensors(records)
    model = SpellProposalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = multitask_loss(model(input_ids), targets)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


def save_checkpoint(model: SpellProposalModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {"embedding_dim": 48, "hidden_dim": 64, "max_length": 96},
        "output_heads": OUTPUT_HEADS,
        "format_version": 1,
    }, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/training-spells.expanded.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/spell_ai.pt"))
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    records = load_records(args.data)
    splits = split_records(records)
    model, history = train_model(splits["train"], epochs=args.epochs)
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
