"""First-answer-character diagnostic using the smallest Transformer candidate."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from syllable_core import SyllableTokenizer, SyllableTransformer


def read_rows(path):
    return [(q.strip(), a.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if "\t" in line for q, a in [line.split("\t", 1)]]


def score(model, data, tok, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for q, a in data:
            x = tok.encode(f"질문: {q} 답변: ").unsqueeze(0).to(device)
            logits = model(x, tok.jamo_ids(x[0].cpu()).unsqueeze(0).to(device))[:, -1]
            total += F.cross_entropy(logits, tok.encode(a[0]).to(device)).item()
    return total / max(1, len(data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--valid", required=True)
    p.add_argument("--prefix", required=True); p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args(); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer(); train, valid = read_rows(args.train), read_rows(args.valid)
    model = SyllableTransformer(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1, 4, 128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001); best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for q, a in train:
            x = tok.encode(f"질문: {q} 답변: ").unsqueeze(0).to(device)
            logits = model(x, tok.jamo_ids(x[0].cpu()).unsqueeze(0).to(device))[:, -1]
            loss = F.cross_entropy(logits, tok.encode(a[0]).to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val = score(model, valid, tok, device)
        if val < best:
            best = val; Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_loss": val}, args.prefix)
        if epoch == 1 or epoch % 20 == 0: print(f"epoch={epoch} val_loss={val:.4f}")


if __name__ == "__main__":
    main()
