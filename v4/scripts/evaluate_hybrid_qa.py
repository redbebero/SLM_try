"""Evaluate factual Transformer + arithmetic specialist routing."""

import argparse
from pathlib import Path

import torch

from arithmetic_router import solve
from syllable_core import SyllableGRU, SyllableTransformer, SyllableTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factual-checkpoint", required=True)
    p.add_argument("--factual-data", required=True)
    p.add_argument("--math-data", required=True)
    args = p.parse_args()
    tok = SyllableTokenizer()
    device = torch.device("cpu")
    model = SyllableTransformer(tok.get_vocab_size(), tok.jamo.get_vocab_sizes(), 16, 64, 1, 4, 128).to(device)
    model.load_state_dict(torch.load(args.factual_checkpoint, map_location=device)["model"])
    model.eval()

    factual_hits = factual_total = 0
    for line in Path(args.factual_data).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        question, answer = line.split("\t", 1)
        prompt = f"질문: {question.strip()} 답변: "
        ids = tok.encode(prompt).unsqueeze(0)
        generated = ""
        for _ in range(len(answer.strip()) + 1):
            with torch.no_grad():
                nxt = model(ids, tok.jamo_ids(ids[0]).unsqueeze(0))[:, -1].argmax(-1, keepdim=True)
            char = tok.decode(nxt[0]); generated += char; ids = torch.cat([ids, nxt], dim=1)
            if char == "\n":
                break
        factual_hits += generated.rstrip("\n") == answer.strip(); factual_total += 1

    math_hits = math_total = 0
    for line in Path(args.math_data).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        question, answer = line.split("\t", 1)
        math_hits += solve(question) == answer.strip(); math_total += 1
    print(f"factual={factual_hits}/{factual_total} math={math_hits}/{math_total} total={factual_hits + math_hits}/{factual_total + math_total}")


if __name__ == "__main__":
    main()
