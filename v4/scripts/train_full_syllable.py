"""Train a compact full-syllable causal SFT decoder."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from full_syllable_lm import FullSyllableCausalLM
from syllable_core import SyllableTokenizer


def encode_record(record, tokenizer):
    prompt = f"Q: {record['question']}|"
    full = tokenizer.encode(prompt + record["answer"] + "\n")
    prompt_len = len(tokenizer.encode(prompt))
    full_mask = torch.ones(len(full), dtype=torch.float32)
    full_mask[:prompt_len] = 0
    return full[:-1], full[1:], full_mask[1:]


def load_records(path, limit=0):
    rows = [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def collate(batch, tokenizer, max_len):
    samples = []
    for record in batch:
        x, y, mask = encode_record(record, tokenizer)
        x, y, mask = x[:max_len], y[:max_len], mask[:max_len]
        samples.append((x, y, mask, tokenizer.jamo_ids(x)))
    length = max(item[0].size(0) for item in samples)
    xs, ys, masks, jamos = [], [], [], []
    for x, y, mask, jamo in samples:
        pad = length - x.size(0)
        xs.append(F.pad(x, (0, pad)))
        ys.append(F.pad(y, (0, pad)))
        masks.append(F.pad(mask, (0, pad)))
        jamos.append(F.pad(jamo, (0, 0, 0, pad)))
    return torch.stack(xs), torch.stack(ys), torch.stack(masks), torch.stack(jamos)


def evaluate(model, rows, tokenizer, device, batch_size, max_len):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x, y, mask, jamo = collate(rows[start:start + batch_size], tokenizer, max_len)
            logits = model(x.to(device), jamo.to(device))
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1), reduction="none"
            ).reshape_as(mask)
            total += (losses * mask.to(device)).sum().item()
            count += mask.sum().item()
    return total / max(1.0, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--valid-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = SyllableTokenizer()
    train_rows = load_records(args.data_dir, args.limit)
    valid_rows = load_records(args.valid_data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullSyllableCausalLM(
        tokenizer.get_vocab_size(), tokenizer.jamo.get_vocab_sizes(),
        args.emb_dim, args.hidden_dim, args.layers, args.heads, args.max_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda batch: collate(batch, tokenizer, args.max_len))
        for x, y, mask, jamo in loader:
            x, y, mask, jamo = x.to(device), y.to(device), mask.to(device), jamo.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, jamo)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).reshape_as(mask)
            loss = (losses * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        valid_loss = evaluate(model, valid_rows, tokenizer, device, args.batch, args.max_len)
        print(f"epoch={epoch} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "vocab": tokenizer.itos,
                        "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim,
                        "layers": args.layers, "heads": args.heads,
                        "max_len": args.max_len}, args.output)


if __name__ == "__main__":
    main()
