"""Masked jamo/track pretraining on downloaded Korean text."""

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hangul_semantic_encoder import HangulSemanticEncoder, count_parameters
from tokenizer import KoJamoTokenizer


def pad_text_batch(texts, tokenizer, max_len, device):
    encoded = [tokenizer.encode(text)[:max_len] for text in texts]
    batch = torch.zeros(len(encoded), max_len, 6, dtype=torch.long, device=device)
    mask = torch.zeros(len(encoded), max_len, dtype=torch.bool, device=device)
    for index, item in enumerate(encoded):
        length = min(item.size(0), max_len)
        if length:
            batch[index, :length] = item[:length].to(device)
            mask[index, :length] = True
    return batch, mask


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KoJamoTokenizer()
    rows = [line.strip() for line in Path(args.text).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    model = HangulSemanticEncoder(
        num_categories=args.num_categories, emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(rows, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda part: pad_text_batch(
                            part, tokenizer, args.max_len, device))
    model.train()
    total, steps = 0.0, 0
    for batch, mask in loader:
        masked = batch.clone()
        selected = (torch.rand(mask.shape, device=device) < args.mask_prob) & mask
        masked[selected] = 0
        logits = model.reconstruct(masked, mask)
        loss = 0.0
        selected_count = selected.sum().clamp_min(1)
        for track, track_logits in enumerate(logits):
            loss = loss + F.cross_entropy(
                track_logits[selected], batch[:, :, track][selected], reduction="sum"
            ) / selected_count
        loss = loss / len(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim, "output_dim": args.output_dim,
        "num_categories": args.num_categories,
        "parameters": count_parameters(model), "rows": len(rows),
    }, output)
    print(f"rows={len(rows)} steps={steps} loss={total / max(1, steps):.4f} "
          f"parameters={count_parameters(model)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--num-categories", type=int, default=6)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    for _ in range(args.epochs):
        run(args)


if __name__ == "__main__":
    main()
