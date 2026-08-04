"""Train the compact joint-output Korean structural Transformer."""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import KoJamoDataset
from train_sft import sft_collate_fn
from structural_transformer import StructuralKoreanTransformer


def masked_ce(logits, targets, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)


def joint_target(y):
    return ((y[:, :, 0] - 1) * 21 * 28
            + (y[:, :, 1] - 1) * 28 + y[:, :, 2]).clamp_min(0)


def loss_for_batch(model, x, y, mask):
    type_logits, char_logits, sym_logits, eng_logits, num_logits = model(x)
    types = model.token_types(y)
    total = masked_ce(type_logits, types, mask)
    korean = mask * (types == 0)
    symbol = mask * (types == 1)
    english = mask * (types == 2)
    number = mask * (types == 3)
    complete = korean * (y[:, :, 0] > 0) * (y[:, :, 1] > 0)
    total = total + masked_ce(char_logits, joint_target(y), complete)
    total = total + masked_ce(sym_logits, y[:, :, 3], symbol)
    total = total + masked_ce(eng_logits, y[:, :, 4], english)
    total = total + masked_ce(num_logits, y[:, :, 5], number)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="train_data_external_mix")
    parser.add_argument("--valid-data-dir", default="train_data_external_mix_valid")
    parser.add_argument("--output", default="checkpoints/structural_transformer_external_best.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = KoJamoDataset(args.data_dir, seq_length=1000, stride=100, is_sft=True)
    valid_data = KoJamoDataset(args.valid_data_dir, seq_length=1000, stride=100, is_sft=True)
    collate = lambda batch: sft_collate_fn(batch, max_seq_length=args.max_seq_length)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=False, collate_fn=collate)
    model = StructuralKoreanTransformer(
        train_data.tokenizer.get_vocab_sizes(), emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim, layers=args.layers, heads=args.heads,
        max_seq_length=args.max_seq_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, x, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item()
        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for x, y, mask in valid_loader:
                valid_total += loss_for_batch(model, x.to(device), y.to(device), mask.to(device)).item()
        train_loss = train_total / max(1, len(train_loader))
        valid_loss = valid_total / max(1, len(valid_loader))
        print(json.dumps({"epoch": epoch, "train": train_loss, "valid": valid_loss}), flush=True)
        if valid_loss < best:
            best = valid_loss
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save({"model": model.state_dict(), "structural_transformer": True,
                        "sft_format": True, "max_seq_length": args.max_seq_length,
                        "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim,
                        "layers": args.layers, "heads": args.heads}, args.output)
            print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
