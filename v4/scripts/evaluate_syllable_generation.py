"""Evaluate free-running syllable GRU on held-out Q/A rows."""
import argparse
import json
from pathlib import Path

import torch

from syllable_core import SyllableGRU, SyllableTokenizer


def generate(model, tokenizer, prompt, device, max_new=120, repetition_penalty=1.0):
    ids = tokenizer.encode(prompt).to(device)
    for _ in range(max_new):
        jamo = tokenizer.jamo_ids(ids.cpu()).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(ids.unsqueeze(0), jamo)[0, -1]
        logits[tokenizer.pad_id] = float("-inf")
        if repetition_penalty > 1.0:
            for token_id in ids.unique():
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        next_id = logits.argmax().view(1)
        ids = torch.cat([ids, next_id])
        if int(next_id) == tokenizer.stoi.get("\n", -1):
            break
    return tokenizer.decode(ids[len(tokenizer.encode(prompt)):].cpu()).rstrip("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()
    tokenizer = SyllableTokenizer()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = checkpoint["model"]
    model = SyllableGRU(
        tokenizer.get_vocab_size(), tokenizer.jamo.get_vocab_sizes(),
        state["embedding.weight"].shape[1], state["proj.bias"].shape[0], 1,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    rows = [json.loads(line) for line in (Path(args.data_dir) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    results = []
    for row in rows:
        prompt = f"Q: {row['question']}|"
        output = generate(model, tokenizer, prompt, device,
                          repetition_penalty=args.repetition_penalty)
        results.append({"question": row["question"], "expected": row["answer"], "output": output,
                        "exact": output == row["answer"], "prefix10": output[:10] == row["answer"][:10]})
    report = {"rows": len(results), "exact": sum(r["exact"] for r in results),
              "prefix10": sum(r["prefix10"] for r in results), "results": results}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
