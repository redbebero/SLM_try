"""Exact-match evaluation for TSV Korean QA pairs."""

import argparse
from pathlib import Path

import torch

from syllable_core import SyllableGRU, SyllableTransformer, SyllableTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--model", choices=("gru", "transformer"), default="gru")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--type-tag", default="")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    tok = SyllableTokenizer()
    if args.model == "transformer":
        model = SyllableTransformer(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1, 4, args.max_len).to(device)
    else:
        model = SyllableGRU(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    hits = total = 0
    for index, raw in enumerate(Path(args.data).read_text(encoding="utf-8").splitlines()):
        if args.limit and index >= args.limit:
            break
        if "\t" not in raw:
            continue
        question, answer = raw.split("\t", 1)
        prefix = f"유형: {args.type_tag} " if args.type_tag else ""
        prompt = f"{prefix}질문: {question.strip()} 답변: "
        ids = tok.encode(prompt).unsqueeze(0).to(device)
        generated = ""
        for _ in range(len(answer.strip()) + 1):
            with torch.no_grad():
                nxt = model(ids, tok.jamo_ids(ids[0].cpu()).unsqueeze(0).to(device))[:, -1].argmax(-1, keepdim=True)
            char = tok.decode(nxt[0])
            generated += char
            ids = torch.cat([ids, nxt], dim=1)
            if char == "\n":
                break
        predicted = generated.rstrip("\n")
        hits += predicted == answer.strip()
        total += 1
        print(f"target={answer.strip()!r} predicted={predicted!r}")
    print(f"exact={hits}/{total}")


if __name__ == "__main__":
    main()
