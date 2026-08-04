"""Pretrain compact HRMContextNet on ordinary Korean text windows."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import KoJamoDataset
from hrm_model import HRMContextNet
from train_hrm import masked_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--valid-data-dir", default=None,
                        help="Disjoint validation directory; avoids overlapping random windows.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--copy-head", action="store_true")
    parser.add_argument("--current-jong", action="store_true")
    parser.add_argument("--char-head", action="store_true")
    parser.add_argument("--joint-jamo", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KoJamoDataset(
        data_dir=args.data_dir, seq_length=args.seq_length,
        stride=args.stride, is_sft=False,
    )
    if args.valid_data_dir:
        train_set = dataset
        val_set = KoJamoDataset(
            data_dir=args.valid_data_dir, seq_length=args.seq_length,
            stride=args.stride, is_sft=False,
        )
    else:
        split = max(1, int(len(dataset) * 0.95))
        train_set, val_set = random_split(
            dataset, [split, len(dataset) - split],
            generator=torch.Generator().manual_seed(42),
        )
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            drop_last=True, pin_memory=True)
    model = HRMContextNet(
        vocab_sizes=dataset.tokenizer.get_vocab_sizes(), emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim, cycle_steps=4, context_layers=1,
        use_copy=args.copy_head, use_current_jong=args.current_jong,
        use_char_head=args.char_head, use_joint_jamo=args.joint_jamo,
        max_seq_length=args.seq_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"device={device} windows={len(dataset)} train={len(train_set)} val={len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for x, y in tqdm(train_loader, desc=f"HRM pretrain {epoch}"):
            if args.limit_batches is not None and batches >= args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            mask = torch.ones(x.size(0), x.size(1), device=device)
            optimizer.zero_grad()
            state = None
            loss = torch.zeros((), device=device)
            for _ in range(args.segments):
                logits, state = model.forward_segment(x, state)
                loss = loss + masked_loss(logits, y, mask) / args.segments
                state = tuple(item.detach() for item in state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for index, (x, y) in enumerate(val_loader):
                if index >= args.val_batches:
                    break
                x, y = x.to(device), y.to(device)
                mask = torch.ones(x.size(0), x.size(1), device=device)
                state = None
                logits = None
                for _ in range(args.segments):
                    logits, state = model.forward_segment(x, state)
                    state = tuple(item.detach() for item in state)
                val_total += masked_loss(logits, y, mask).item()
        val = val_total / max(1, min(len(val_loader), args.val_batches))
        print(f"HRM pretrain epoch={epoch} train={total / max(1,batches):.4f} val={val:.4f}")
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "hrm_segments": args.segments,
                        "context_encoder": True}, args.output)
            print(f"best saved: {args.output}")


if __name__ == "__main__":
    main()
