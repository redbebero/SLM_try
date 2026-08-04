"""Train a small bidirectional Transformer extractive QA span model."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from syllable_core import SyllableBiDAF, SyllableSpanTransformer, SyllableTokenizer


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--valid", required=True); p.add_argument("--prefix", required=True); p.add_argument("--epochs", type=int, default=5); p.add_argument("--hidden", type=int, default=64); p.add_argument("--layers", type=int, default=1); p.add_argument("--pretrained", default=""); p.add_argument("--model", choices=("span", "bidaf"), default="span")
    args = p.parse_args(); torch.manual_seed(42); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer(); train, valid = read(args.train), read(args.valid)
    model_class = SyllableBiDAF if args.model == "bidaf" else SyllableSpanTransformer
    core = model_class(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, args.hidden, args.layers, 4, 256).to(device)
    if args.pretrained:
        source = torch.load(args.pretrained, map_location=device)["model"]
        target = core.state_dict()
        compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
        if args.model == "bidaf":
            compatible = {f"encoder.{key}": value for key, value in source.items() if f"encoder.{key}" in target and target[f"encoder.{key}"].shape == value.shape}
        core.load_state_dict(compatible, strict=False)
        print(f"transferred={len(compatible)}")
    opt = torch.optim.AdamW(core.parameters(), lr=0.001)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        core.train(); total = 0.0
        for row in train:
            ids = tok.encode(row["prompt"]).unsqueeze(0).to(device)
            split = row["prompt"].find(" 질문:") + 1
            segments = torch.tensor([[0 if i < split else 1 for i in range(len(row["prompt"]))]], device=device)
            start_logits, end_logits = core(ids, tok.jamo_ids(ids[0].cpu()).unsqueeze(0).to(device), segments)
            loss = F.cross_entropy(start_logits, torch.tensor([row["start"]], device=device)) + F.cross_entropy(end_logits, torch.tensor([row["end"]], device=device))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); total += loss.item()
        val = validate(core, tok, valid, device)
        print(f"epoch={epoch} train_loss={total / max(1, len(train)):.4f} val_exact={val:.4f}")
        if 1.0 - val < best:
            best = 1.0 - val; Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"core": core.state_dict()}, args.prefix)


@torch.no_grad()
def validate(core, tok, rows, device):
    core.eval(); hits = 0
    for row in rows:
        ids = tok.encode(row["prompt"]).unsqueeze(0).to(device)
        split = row["prompt"].find(" 질문:") + 1
        segments = torch.tensor([[0 if i < split else 1 for i in range(len(row["prompt"]))]], device=device)
        start_logits, end_logits = core(ids, tok.jamo_ids(ids[0].cpu()).unsqueeze(0).to(device), segments)
        s, e = start_logits[0].argmax().item(), end_logits[0].argmax().item()
        if s <= e and tok.decode(ids[0, s:e + 1].cpu()) == row["answer"]: hits += 1
    return hits / max(1, len(rows))


if __name__ == "__main__":
    main()
