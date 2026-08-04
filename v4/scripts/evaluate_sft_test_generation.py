"""Evaluate free-running generation on real held-out SFT records."""
import argparse
import json
from pathlib import Path

import torch

from chat import generate, load_model
from tokenizer import KoJamoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (Path(args.data_dir) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    tokenizer = KoJamoTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, tokenizer.get_vocab_sizes(), device=device)
    results = []
    for row in rows:
        prompt = f"Q: {row['question']}\nA: "
        output = generate(model, tokenizer, prompt, max_new_chars=120, device=device,
                          stop_on_newline=True, use_reasoning_router=False,
                          allow_refusal=False)
        results.append({
            "question": row["question"], "expected": row["answer"], "output": output,
            "exact": output == row["answer"],
            "prefix10": output[:10] == row["answer"][:10],
        })
    report = {
        "rows": len(results),
        "exact": sum(row["exact"] for row in results),
        "prefix10": sum(row["prefix10"] for row in results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
