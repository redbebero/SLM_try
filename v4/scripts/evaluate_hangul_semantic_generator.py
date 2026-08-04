"""Free-running evaluation for the semantic-vector-conditioned generator."""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_generation import load_cases, score_generation
from hangul_semantic_generator import HangulSemanticGenerator
from tokenizer import KoJamoTokenizer
from train_hangul_semantic import load_records, pad_batch


def generate(model, tokenizer, question, device, max_new):
    model.eval()
    source, source_mask = pad_batch([question], tokenizer, model.max_len, device)
    decoder = torch.zeros(1, 1, 6, dtype=torch.long, device=device)
    newline_id = tokenizer.sym_vocab["\n"]
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(source, source_mask, decoder)
            next_step = torch.stack([
                head[:, -1].argmax(dim=-1) for head in logits
            ], dim=-1)
            decoder = torch.cat([decoder, next_step.unsqueeze(1)], dim=1)
            if int(next_step[0, 3]) == newline_id:
                break
    return tokenizer.decode(decoder[0, 1:].cpu()).split("\n", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", default="eval/ko_generation_cases.jsonl")
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = HangulSemanticGenerator().to(device)
    model.load_state_dict(checkpoint["model"])
    model.max_len = checkpoint.get("max_len", 96)
    tokenizer = KoJamoTokenizer()
    rows = load_cases(args.cases)
    results = []
    for row in rows:
        output = generate(model, tokenizer, row["prompt"], device, args.max_new)
        results.append({"id": row["id"], "prompt": row["prompt"],
                        "output": output,
                        **score_generation(output, row["expected_keywords"])})
    report = {
        "cases": len(results),
        "keyword_hit_rate": sum(r["keyword_hit_rate"] for r in results) / len(results),
        "mean_hangul_ratio": sum(r["full_hangul_ratio"] for r in results) / len(results),
        "rows": results,
    }
    if args.test_dir:
        downloaded = load_records(args.test_dir)
        if args.limit:
            downloaded = downloaded[:args.limit]
        report["downloaded_holdout"] = []
        for row in downloaded:
            output = generate(model, tokenizer, row["question"], device, args.max_new)
            report["downloaded_holdout"].append({
                "question": row["question"], "answer": row["answer"],
                "output": output,
            })
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "rows" and key != "downloaded_holdout"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
