"""Train the small character GRU on Korean question-answer pairs."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from syllable_core import SyllableGRU, SyllableTransformer, SyllableTokenizer


def rows(path):
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            question, answer = line.split("\t", 1)
            result.append((question.strip(), answer.strip()))
    return result


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer()
    train_rows = rows(args.train)
    val_rows = rows(args.valid)
    model_class = SyllableTransformer if args.model == "transformer" else SyllableGRU
    model = model_class(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), args.emb, args.hidden, 1, 4, 128).to(device) if args.model == "transformer" else model_class(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), args.emb, args.hidden, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for question, answer in train_rows:
            text = f"질문: {question} 답변: {answer}\n"
            answer_start = text.index("답변: ") + len("답변: ")
            ids = tok.encode(text)
            x, y = ids[:-1].unsqueeze(0).to(device), ids[1:].unsqueeze(0).to(device)
            jamo = tok.jamo_ids(x[0]).unsqueeze(0).to(device)
            logits = model(x, jamo)
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            mask = torch.zeros_like(losses)
            mask[answer_start - 1:] = 1.0
            loss = (losses * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        val_loss = validation_loss(model, val_rows, tok, device)
        print(f"epoch={epoch} train_loss={total / max(1, len(train_rows)):.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, args.prefix)


@torch.no_grad()
def validation_loss(model, data, tok, device):
    model.eval()
    total = 0.0
    for question, answer in data:
        text = f"질문: {question} 답변: {answer}\n"
        start = text.index("답변: ") + len("답변: ")
        ids = tok.encode(text)
        x, y = ids[:-1].unsqueeze(0).to(device), ids[1:].unsqueeze(0).to(device)
        logits = model(x, tok.jamo_ids(x[0]).unsqueeze(0).to(device))
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        total += losses[start - 1:].mean().item()
    return total / max(1, len(data))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--emb", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=("gru", "transformer"), default="gru")
    train(p.parse_args())
