"""Compact causal language-model pretraining on Korean text."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from subword_lm import CompactSubwordCausalLM
from train_subword_lm import SubwordTokenizer


def chunk_ids(ids, max_len):
    """Return next-token chunks, each containing at most max_len+1 ids."""
    return [ids[start:start + max_len + 1] for start in range(0, max(0, len(ids) - 1), max_len)]


def load_chunks(path, tokenizer, max_len, limit=0, start=0):
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if start:
        rows = rows[start:]
    if limit:
        rows = rows[:limit]
    chunks = []
    for row in rows:
        if not row.strip():
            continue
        ids = tokenizer.encode(row + "<eos>")
        chunks.extend(chunk_ids(ids, max_len))
    return [chunk for chunk in chunks if len(chunk) > 1]


def load_record_chunks(path, tokenizer, max_len, limit=0, start=0):
    rows = [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[start:]
    if limit:
        rows = rows[:limit]
    chunks = []
    for row in rows:
        ids = tokenizer.encode(f"Q: {row['question']}<answer>{row['answer']}<eos>")
        chunks.extend(chunk_ids(ids, max_len))
    return [chunk for chunk in chunks if len(chunk) > 1]


def collate(chunks, pad_id):
    length = max(len(chunk) - 1 for chunk in chunks)
    xs, ys, masks = [], [], []
    for ids in chunks:
        x, y = ids[:-1], ids[1:]
        pad = length - len(x)
        xs.append(F.pad(x, (0, pad), value=pad_id))
        ys.append(F.pad(y, (0, pad), value=pad_id))
        masks.append(F.pad(torch.ones_like(y, dtype=torch.float32), (0, pad)))
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


def evaluate(model, chunks, tokenizer, device, batch, max_len):
    model.eval()
    total = count = 0.0
    with torch.no_grad():
        loader = DataLoader(chunks, batch_size=batch, shuffle=False,
                            collate_fn=lambda rows: collate(rows, tokenizer.pad_id))
        for x, y, mask in loader:
            logits = model(x.to(device))
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1), reduction="none").reshape_as(mask)
            active = mask.to(device) * (y.to(device) != tokenizer.pad_id)
            total += (losses * active).sum().item()
            count += active.sum().item()
    return total / max(1.0, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--valid-limit", type=int, default=2000)
    parser.add_argument("--valid-start", type=int, default=0)
    parser.add_argument("--records-dir", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    tokenizer = SubwordTokenizer(args.tokenizer)
    loader = load_record_chunks if args.records_dir else load_chunks
    train = loader(args.train_file, tokenizer, args.max_len, args.limit)
    valid = loader(args.valid_file, tokenizer, args.max_len, args.valid_limit, args.valid_start)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactSubwordCausalLM(tokenizer.vocab_size(), args.emb_dim, args.hidden_dim,
                                   args.layers, args.heads, args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda rows: collate(rows, tokenizer.pad_id))
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").reshape_as(mask)
            active = mask * (y != tokenizer.pad_id)
            loss = (losses * active).sum() / active.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        valid_loss = evaluate(model, valid, tokenizer, device, args.batch, args.max_len)
        print(f"epoch={epoch} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "tokenizer": str(args.tokenizer),
                        "vocab_size": tokenizer.vocab_size(), "emb_dim": args.emb_dim,
                        "hidden_dim": args.hidden_dim, "layers": args.layers,
                        "heads": args.heads, "max_len": args.max_len,
                        "answer_id": tokenizer.answer_id, "eos_id": tokenizer.eos_id}, args.output)


if __name__ == "__main__":
    main()
