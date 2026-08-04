"""Tiny character-level GRU baseline for the compositional toy corpus."""

from pathlib import Path

import torch
from torch import nn


class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        h, hidden = self.gru(self.embedding(x), hidden)
        return self.head(h), hidden


def read_lines(path):
    return [line.rstrip("\n") + "\n" for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    torch.manual_seed(42)
    train = read_lines("experiments/compositional_toy/train/train.txt")
    valid = read_lines("experiments/compositional_toy/valid/valid.txt")
    chars = sorted(set("".join(train + valid)))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    encode = lambda text: torch.tensor([stoi[char] for char in text], dtype=torch.long)

    model = CharGRU(len(stoi)).cuda() if torch.cuda.is_available() else CharGRU(len(stoi))
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_ids = [encode(line).to(device) for line in train]
    valid_ids = [encode(line).to(device) for line in valid]

    for epoch in range(1, 121):
        model.train()
        total = 0.0
        for ids in train_ids:
            optimizer.zero_grad()
            logits, _ = model(ids[:-1].unsqueeze(0))
            loss = loss_fn(logits.reshape(-1, len(stoi)), ids[1:])
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch in (1, 20, 40, 60, 80, 100, 120):
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    loss_fn(model(ids[:-1].unsqueeze(0))[0].reshape(-1, len(stoi)), ids[1:]).item()
                    for ids in valid_ids
                ) / len(valid_ids)
            print(f"epoch={epoch} train_loss={total / len(train_ids):.4f} valid_loss={val_loss:.4f}")

    def generate(prompt, max_new=12):
        model.eval()
        ids = encode(prompt).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, hidden = model(ids)
            current = logits[:, -1].argmax(dim=-1, keepdim=True)
            output = prompt
            for _ in range(max_new):
                char = itos[int(current.item())]
                output += char
                if char == "\n":
                    break
                logits, hidden = model(current, hidden)
                current = logits[:, -1].argmax(dim=-1, keepdim=True)
        return output.rstrip("\n")

    exact = 0
    for line in valid:
        prompt, answer = line.rsplit(" ", 1)
        predicted = generate(prompt + " ")
        expected = line.rstrip("\n")
        exact += predicted == expected
        print(f"prompt={prompt!r} predicted={predicted!r} expected={expected!r}")
    print(f"valid_exact={exact}/{len(valid)}")
    torch.save({"model": model.state_dict(), "stoi": stoi}, "checkpoints/verify_char_compositional.pth")


if __name__ == "__main__":
    main()
