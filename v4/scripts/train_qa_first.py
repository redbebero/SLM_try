"""Diagnose whether the small GRU can select the first answer character."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from syllable_core import SyllableGRU, SyllableTokenizer


def read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            q, a = line.split("\t", 1)
            if a.strip():
                rows.append((q.strip(), a.strip()))
    return rows


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer()
    train, valid = read_rows(args.train), read_rows(args.valid)
    model = SyllableGRU(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for question, answer in train:
            prompt = tok.encode(f"질문: {question} 답변: ").unsqueeze(0).to(device)
            jamo = tok.jamo_ids(prompt[0].cpu()).unsqueeze(0).to(device)
            logits = model(prompt, jamo)[:, -1]
            target = tok.encode(answer[0]).to(device)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        val_loss = score_loss(model, valid, tok, device)
        if val_loss < best:
            best = val_loss
            Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, args.prefix)
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch} train_loss={total / len(train):.4f} val_loss={val_loss:.4f}")


@torch.no_grad()
def score_loss(model, rows, tok, device):
    model.eval()
    total = 0.0
    for question, answer in rows:
        prompt = tok.encode(f"질문: {question} 답변: ").unsqueeze(0).to(device)
        logits = model(prompt, tok.jamo_ids(prompt[0].cpu()).unsqueeze(0).to(device))[:, -1]
        total += F.cross_entropy(logits, tok.encode(answer[0]).to(device)).item()
    return total / max(1, len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())
