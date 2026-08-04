"""Train the small jamo-aware GRU on unrestricted character sequences."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from syllable_core import SyllableGRU, SyllableTransformer, SyllableTokenizer, read_lines


class CharWindowDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=32, stride=32, max_windows=0):
        windows = []
        for line in read_lines(path):
            ids = tokenizer.encode(line + "\n")
            if len(ids) <= seq_len:
                continue
            for start in range(0, len(ids) - seq_len, stride):
                windows.append((ids[start:start + seq_len], ids[start + 1:start + seq_len + 1]))
        if max_windows and len(windows) > max_windows:
            step = len(windows) / max_windows
            windows = [windows[int(index * step)] for index in range(max_windows)]
        self.windows = windows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x, y = self.windows[index]
        return x, y, self.tokenizer.jamo_ids(x)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for x, y, jamo in loader:
        logits = model(x.to(device), jamo.to(device))
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).item()
        count += 1
    model.train()
    return total / max(1, count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--prefix", default="checkpoints/char_full_best.pth")
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=5000)
    p.add_argument("--val-max-windows", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--emb", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=("gru", "transformer"), default="gru")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer()
    train = CharWindowDataset(args.data, tok, args.seq, args.stride, args.max_windows)
    valid = CharWindowDataset(args.val_data, tok, args.seq, args.stride, args.val_max_windows)
    if not train or not valid:
        raise ValueError(f"empty dataset: train={len(train)} valid={len(valid)}")
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=args.batch, shuffle=False)
    if args.model == "transformer":
        model = SyllableTransformer(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), args.emb, args.hidden, args.layers, 4, 128).to(device)
    else:
        model = SyllableGRU(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), args.emb, args.hidden, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    print(f"char_full device={device} train_windows={len(train)} val_windows={len(valid)} vocab={tok.get_vocab_size()}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y, jamo in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), jamo.to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        val_loss = evaluate(model, valid_loader, device)
        train_loss = total / len(train_loader)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, args.prefix)


if __name__ == "__main__":
    main()
