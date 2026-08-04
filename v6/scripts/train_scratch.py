import argparse
import json
from pathlib import Path

import torch
from torch import nn

try:
    from scripts.train_lora import format_record
except ModuleNotFoundError:
    from train_lora import format_record


class ScratchGRU(nn.Module):
    def __init__(self, vocab_size, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.head(self.gru(self.embedding(x))[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [format_record(json.loads(line)) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    chars = sorted(set("".join(texts)))
    vocab = {char: index + 1 for index, char in enumerate(chars)}
    pad = 0
    encoded = [torch.tensor([vocab[char] for char in text[:256]], dtype=torch.long) for text in texts]
    model = ScratchGRU(len(vocab) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for start in range(0, len(encoded), 8):
            batch = encoded[start : start + 8]
            width = max(len(item) for item in batch)
            inputs = torch.full((len(batch), width - 1), pad, dtype=torch.long, device=device)
            targets = torch.full_like(inputs, pad)
            for row, item in enumerate(batch):
                inputs[row, : len(item) - 1] = item[:-1].to(device)
                targets[row, : len(item) - 1] = item[1:].to(device)
            loss = loss_fn(model(inputs).transpose(1, 2), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch + 1} loss={total:.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab, "pad": pad, "hidden": 128}, output)
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters())}")


if __name__ == "__main__":
    main()
