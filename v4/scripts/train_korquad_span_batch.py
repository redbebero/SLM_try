"""Batched trainer for the small Korean extractive span model."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from syllable_core import (SyllableBiaffineSpanTransformer, SyllableCrossSpanTransformer,
                            SyllableDoubleCrossSpanTransformer, SyllablePointerSpanTransformer,
                            SyllableSpanTransformer, SyllableTokenizer)


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def collate(rows, tok, device):
    encoded = [tok.encode(row["prompt"]) for row in rows]
    length = max(x.numel() for x in encoded)
    ids = torch.zeros(len(rows), length, dtype=torch.long)
    jamo = torch.zeros(len(rows), length, 3, dtype=torch.long)
    segments = torch.ones(len(rows), length, dtype=torch.long)
    padding = torch.ones(len(rows), length, dtype=torch.bool)
    starts, ends = [], []
    for n, (row, item) in enumerate(zip(rows, encoded)):
        size = item.numel()
        ids[n, :size] = item
        # KoJamoTokenizer returns six feature columns; the model uses cho/jung/jong.
        jamo[n, :size] = tok.jamo_ids(item)[:, :3]
        split = row["prompt"].find(" 질문:") + 1
        segments[n, :split] = 0
        padding[n, :size] = False
        starts.append(row["start"])
        ends.append(row["end"])
    return (ids.to(device), jamo.to(device), segments.to(device), padding.to(device),
            torch.tensor(starts, device=device), torch.tensor(ends, device=device))


def context_logits(start_logits, end_logits, segments, padding):
    forbidden = segments.eq(1) | padding
    return start_logits.masked_fill(forbidden, -1e9), end_logits.masked_fill(forbidden, -1e9)


def best_span(start_logits, end_logits, max_span=64):
    length = start_logits.size(1)
    index = torch.arange(length, device=start_logits.device)
    valid = index[None, :, None] <= index[None, None, :]
    valid = valid & ((index[None, None, :] - index[None, :, None]) < max_span)
    scores = start_logits[:, :, None] + end_logits[:, None, :]
    return scores.masked_fill(~valid, -1e9).flatten(1).argmax(1)


def pair_mask(segments, padding):
    allowed = segments.eq(0) & ~padding
    return allowed[:, :, None] & allowed[:, None, :]


@torch.no_grad()
def validate(core, tok, rows, device, batch_size):
    core.eval(); hits = 0
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate(x, tok, device))
    for batch, part in zip(loader, [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]):
        ids, jamo, segments, padding, _, _ = batch
        if hasattr(core, "span_logits"):
            scores = core.span_logits(ids, jamo, segments, padding)
            scores = scores.masked_fill(~pair_mask(segments, padding), -1e9)
            flat_spans = scores.flatten(1).argmax(1)
        else:
            start_logits, end_logits = core(ids, jamo, segments, padding)
            start_logits, end_logits = context_logits(start_logits, end_logits, segments, padding)
            flat_spans = best_span(start_logits, end_logits)
        for n, row in enumerate(part):
            s, e = divmod(flat_spans[n].item(), ids.size(1))
            if tok.decode(ids[n, s:e + 1].cpu()) == row["answer"]:
                hits += 1
    return hits / max(1, len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--valid", required=True)
    p.add_argument("--prefix", required=True); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1); p.add_argument("--pretrained", default="")
    p.add_argument("--model", choices=("span", "cross", "biaffine", "double", "pointer"), default="span")
    args = p.parse_args(); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SyllableTokenizer(); train, valid = read(args.train), read(args.valid)
    model_class = {"span": SyllableSpanTransformer, "cross": SyllableCrossSpanTransformer,
                   "biaffine": SyllableBiaffineSpanTransformer,
                   "double": SyllableDoubleCrossSpanTransformer,
                   "pointer": SyllablePointerSpanTransformer}[args.model]
    core = model_class(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, args.hidden, args.layers, 4, 256).to(device)
    if args.pretrained:
        loaded = torch.load(args.pretrained, map_location=device)
        source = loaded.get("model", loaded.get("core", loaded))
        target = core.state_dict()
        compatible = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
        core.load_state_dict(compatible, strict=False)
        print(f"device={device} transferred={len(compatible)} train={len(train)} valid={len(valid)}")
    else:
        print(f"device={device} train={len(train)} valid={len(valid)}")
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda x: collate(x, tok, device))
    opt = torch.optim.AdamW(core.parameters(), lr=0.001)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        core.train(); total = 0.0
        for ids, jamo, segments, padding, starts, ends in loader:
            if hasattr(core, "span_logits"):
                scores = core.span_logits(ids, jamo, segments, padding)
                scores = scores.masked_fill(~pair_mask(segments, padding), -1e9)
                targets = starts * ids.size(1) + ends
                loss = F.cross_entropy(scores.flatten(1), targets)
            else:
                start_logits, end_logits = core(ids, jamo, segments, padding)
                start_logits, end_logits = context_logits(start_logits, end_logits, segments, padding)
                loss = F.cross_entropy(start_logits, starts) + F.cross_entropy(end_logits, ends)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step(); total += loss.item()
        val = validate(core, tok, valid, device, args.batch_size)
        print(f"epoch={epoch} train_loss={total / max(1, len(loader)):.4f} val_exact={val:.4f}")
        if val > best:
            best = val; Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"core": core.state_dict()}, args.prefix)


if __name__ == "__main__":
    main()
