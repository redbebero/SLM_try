"""Train the semantic-vector-conditioned Hangul generator."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hangul_semantic_generator import HangulSemanticGenerator, load_semantic_encoder
from tokenizer import KoJamoTokenizer
from train_hangul_semantic import load_records, pad_batch


def encode_answers(rows, tokenizer, max_len, device):
    values = [tokenizer.encode(row["answer"] + "\n")[:max_len] for row in rows]
    batch = torch.zeros(len(values), max_len, 6, dtype=torch.long, device=device)
    mask = torch.zeros(len(values), max_len, dtype=torch.bool, device=device)
    for index, value in enumerate(values):
        length = min(value.size(0), max_len)
        if length:
            batch[index, :length] = value[:length].to(device)
            mask[index, :length] = True
    decoder = torch.zeros_like(batch)
    decoder[:, 1:] = batch[:, :-1]
    return decoder, batch, mask


def loss_for(logits, target, mask):
    loss = 0.0
    for index, values in enumerate(logits):
        loss = loss + F.cross_entropy(
            values.reshape(-1, values.size(-1)), target[:, :, index].reshape(-1),
            reduction="none",
        ).masked_select(mask.reshape(-1)).mean()
    return loss / len(logits)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KoJamoTokenizer()
    train_rows = load_records(args.train_dir)
    valid_rows = load_records(args.valid_dir)
    if args.limit:
        train_rows = train_rows[:args.limit]
    model = HangulSemanticGenerator(
        emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim,
    ).to(device)
    if args.init:
        load_semantic_encoder(model, torch.load(args.init, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda rows: rows)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for rows in loader:
            questions = [row["question"] for row in rows]
            source, source_mask = pad_batch(questions, tokenizer, args.max_len, device)
            decoder, target, target_mask = encode_answers(rows, tokenizer, args.max_len, device)
            logits = model(source, source_mask, decoder)
            loss = loss_for(logits, target, target_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        mean_loss = total / max(1, len(loader))
        print(f"epoch={epoch} train_loss={mean_loss:.4f}")
        if mean_loss < best:
            best = mean_loss
            torch.save({"model": model.state_dict(), "max_len": args.max_len}, output)
    print(json.dumps({"train_rows": len(train_rows), "valid_rows": len(valid_rows),
                      "best_train_loss": best}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--init", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
