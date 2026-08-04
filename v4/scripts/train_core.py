"""Minimal trainer for the active small Korean model path."""

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import KoJamoDataset
from model import KoJamoNet


class PromptLineDataset(Dataset):
    """One full line per sample; loss begins at the answer after the last space."""
    def __init__(self, data_dir):
        self.tokenizer = KoJamoDataset(data_dir, seq_length=1, stride=1).tokenizer
        self.samples = []
        self.answer_texts = []
        for path in sorted(Path(data_dir).glob("*.txt")):
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or " " not in line:
                    continue
                prompt_end = line.rfind(" ") + 1
                encoded = self.tokenizer.encode(line + "\n")
                self.samples.append((encoded[:-1], encoded[1:], prompt_end - 1))
                self.answer_texts.append(line[prompt_end:])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, answer_start = self.samples[index]
        mask = torch.zeros(y.shape[0], dtype=torch.float32)
        mask[answer_start:] = 1.0
        return x, y, mask

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_track_loss(logits, target, mask):
    loss = F.nll_loss(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
    )
    return (loss * mask.reshape(-1)).sum() / mask.sum().clamp_min(1)


def batch_loss(model, x, y, active_mask=None):
    outputs = model(x, target_for_forcing=y, teacher_forcing_ratio=1.0)
    type_logits, cho, jung, jong, sym, eng, num = outputs
    types = model._get_types(y)
    if active_mask is None:
        active_mask = torch.ones_like(types, dtype=torch.float32)
    type_loss = F.nll_loss(type_logits.reshape(-1, 4), types.reshape(-1), reduction="none")
    type_loss = (type_loss * active_mask.reshape(-1)).sum() / active_mask.sum().clamp_min(1)
    hangul = (types == 0).float()
    symbol = (types == 1).float()
    english = (types == 2).float()
    number = (types == 3).float()
    losses = [
        masked_track_loss(cho, y[:, :, 0], hangul * active_mask),
        masked_track_loss(jung, y[:, :, 1], hangul * active_mask),
        masked_track_loss(jong, y[:, :, 2], hangul * active_mask),
        masked_track_loss(sym, y[:, :, 3], symbol * active_mask),
        masked_track_loss(eng, y[:, :, 4], english * active_mask),
        masked_track_loss(num, y[:, :, 5], number * active_mask),
    ]
    return type_loss + sum(losses)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x, y = batch[:2]
        loss = batch_loss(model, x.to(device), y.to(device), batch[2].to(device) if len(batch) == 3 else None)
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


def save_checkpoint(path, model, epoch, val_loss):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, path)


def main():
    parser = argparse.ArgumentParser(description="Train the active conditional-jamo GRU.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--prefix", default="checkpoints/core_best.pth")
    parser.add_argument("--emb", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--prompt-completion", action="store_true")
    parser.add_argument("--balance-answers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PromptLineDataset(args.data_dir) if args.prompt_completion else KoJamoDataset(args.data_dir, seq_length=args.seq, stride=args.stride)
    val_set = PromptLineDataset(args.val_data_dir) if args.prompt_completion else KoJamoDataset(args.val_data_dir, seq_length=args.seq, stride=args.seq)
    if not len(train_set) or not len(val_set):
        raise ValueError(f"empty dataset window: train={len(train_set)} val={len(val_set)}")
    sampler = None
    shuffle = True
    if args.prompt_completion and args.balance_answers:
        answers = train_set.answer_texts
        counts = Counter(answers)
        weights = torch.tensor([1.0 / counts[answer] for answer in answers], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train_set, batch_size=1 if args.prompt_completion else args.batch,
        shuffle=shuffle, sampler=sampler, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=1 if args.prompt_completion else args.batch,
        shuffle=False, drop_last=False,
    )
    model = KoJamoNet(
        train_set.tokenizer.get_vocab_sizes(), args.emb, args.hidden, args.layers,
        cascade=True, conditional_decoder=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")

    print(f"active_core device={device} train_windows={len(train_set)} val_windows={len(val_set)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            x, y = batch[:2]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, x.to(device), y.to(device), batch[2].to(device) if len(batch) == 3 else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item()
        val_loss = evaluate(model, val_loader, device)
        train_loss = train_total / max(1, len(train_loader))
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(args.prefix, model, epoch, val_loss)
            print(f"best={args.prefix}")


if __name__ == "__main__":
    main()
