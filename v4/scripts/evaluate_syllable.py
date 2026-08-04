"""Exact evaluation for the syllable-ID GRU."""

import argparse
from pathlib import Path

import torch

from syllable_core import SyllableGRU, SyllableTokenizer, read_lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    tok = SyllableTokenizer()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SyllableGRU(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    hits = 0
    total = 0
    for line in read_lines(args.data):
        if " " not in line:
            continue
        cut = line.rfind(" ") + 1
        prompt = line[:cut]
        answer = line[cut:]
        ids = tok.encode(prompt).unsqueeze(0).to(device)
        generated = prompt
        for _ in range(len(answer) + 1):
            jamo = tok.jamo_ids(ids[0].cpu()).unsqueeze(0).to(device)
            with torch.no_grad():
                next_id = model(ids, jamo)[:, -1].argmax(dim=-1, keepdim=True)
            char = tok.decode(next_id[0])
            generated += char
            ids = torch.cat([ids, next_id], dim=1)
            if char == "\n":
                break
        predicted = generated[len(prompt):].rstrip("\n")
        hits += predicted == answer
        total += 1
    print(f"exact={hits}/{total}")


if __name__ == "__main__":
    main()
