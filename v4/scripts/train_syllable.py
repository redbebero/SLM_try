"""Train the active syllable-ID GRU on prompt-completion samples."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from syllable_core import SyllableGRU, SyllableTokenizer, read_lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-data-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--emb", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delimiter", default="|",
                   help="Prompt/answer delimiter in each downloaded source line.")
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SyllableTokenizer()
    train = read_lines(str(next(Path(args.data_dir).glob("*.txt"))))
    valid = read_lines(str(next(Path(args.val_data_dir).glob("*.txt"))))
    model = SyllableGRU(
        tokenizer.get_vocab_size(), tokenizer.jamo.get_vocab_sizes(),
        args.emb, args.hidden, args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for line in train:
            if args.delimiter not in line:
                continue
            cut = line.rfind(args.delimiter) + 1
            full = tokenizer.encode(line + "\n")
            ids = full[:-1].unsqueeze(0).to(device)
            target = full[1:].to(device)
            jamo = tokenizer.jamo_ids(ids[0].cpu()).unsqueeze(0).to(device)
            start = cut - 1
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, jamo)[0]
            loss = F.cross_entropy(logits[start:], target[start:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            count = 0
            for line in valid:
                if args.delimiter not in line:
                    continue
                cut = line.rfind(args.delimiter) + 1
                full = tokenizer.encode(line + "\n")
                ids = full[:-1].unsqueeze(0).to(device)
                target = full[1:].to(device)
                jamo = tokenizer.jamo_ids(ids[0].cpu()).unsqueeze(0).to(device)
                logits = model(ids, jamo)[0]
                val_total += F.cross_entropy(logits[cut - 1:], target[cut - 1:]).item()
                count += 1
            val_loss = val_total / max(1, count)
        print(f"epoch={epoch} train_loss={total / max(1, len(train)):.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "vocab": tokenizer.itos}, args.prefix)


if __name__ == "__main__":
    main()
