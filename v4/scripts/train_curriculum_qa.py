"""Curriculum + replay trainer for small heterogeneous Korean QA."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from syllable_core import SyllableTransformer, SyllableTokenizer


def read_rows(path, kind):
    return [(kind, q.strip(), a.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if "\t" in line for q, a in [line.split("\t", 1)]]


def loss_one(model, tok, kind, question, answer, device):
    text = f"유형: {kind} 질문: {question} 답변: {answer}\n"
    start = text.index("답변: ") + len("답변: ")
    ids = tok.encode(text)
    x, y = ids[:-1].unsqueeze(0).to(device), ids[1:].unsqueeze(0).to(device)
    logits = model(x, tok.jamo_ids(x[0].cpu()).unsqueeze(0).to(device))
    values = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return values[start - 1:].mean()


def train_phase(model, tok, rows, replay, epochs, device, seed):
    rng = random.Random(seed)
    for _ in range(epochs):
        batch = list(rows) + list(replay)
        rng.shuffle(batch)
        for kind, question, answer in batch:
            loss = loss_one(model, tok, kind, question, answer, device)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.optimizer.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual-train", required=True); p.add_argument("--factual-valid", required=True)
    p.add_argument("--math-train", required=True); p.add_argument("--math-valid", required=True)
    p.add_argument("--prefix", required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--factual-epochs", type=int, default=20)
    p.add_argument("--math-epochs", type=int, default=2)
    p.add_argument("--mixed-epochs", type=int, default=5)
    args = p.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer()
    factual = read_rows(args.factual_train, "사실"); math = read_rows(args.math_train, "수학")
    factual = factual[:180]; math = math[:500]
    model = SyllableTransformer(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1, 4, 256).to(device)
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    print(f"curriculum device={device} factual={len(factual)} math={len(math)}")
    train_phase(model, tok, factual, [], args.factual_epochs, device, args.seed)
    torch.save({"model": model.state_dict(), "phase": "factual"}, f"{args.prefix}.factual")
    train_phase(model, tok, math, factual, args.math_epochs, device, args.seed + 1)
    torch.save({"model": model.state_dict(), "phase": "math_replay"}, f"{args.prefix}.math")
    train_phase(model, tok, factual + math[:180], factual + math[:180], args.mixed_epochs, device, args.seed + 2)
    Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "factual_count": len(factual), "math_count": len(math)}, args.prefix)


if __name__ == "__main__":
    main()
