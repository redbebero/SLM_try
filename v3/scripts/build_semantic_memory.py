"""Build a compact multilingual-E5 index for KorQuAD QA retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    records = []
    for block in args.memory.read_text(encoding="utf-8").split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0].startswith("Q: ") and lines[1].startswith("A: "):
            records.append((lines[0][3:].strip(), lines[1][3:].strip()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch):
            questions = ["passage: " + q for q, _ in records[start:start + args.batch]]
            batch = tokenizer(questions, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            pooled = mean_pool(model(**batch).last_hidden_state, batch["attention_mask"])
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), np.concatenate(vectors))
    args.output.with_suffix(".json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"records": len(records), "vectors": str(args.output.with_suffix('.npy'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
