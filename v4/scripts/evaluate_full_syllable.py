"""Free-running evaluation for FullSyllableCausalLM."""
import argparse
import json
from pathlib import Path

import torch

from full_syllable_lm import FullSyllableCausalLM
from syllable_core import SyllableTokenizer


def generation_context(ids, max_len):
    """Use a fixed-size recent context for arbitrarily long generations."""
    return ids[-max_len:]


def generate(model, tokenizer, prompt, device, max_new=120, repetition_penalty=1.0):
    prompt_ids = tokenizer.encode(prompt)
    ids = prompt_ids.to(device)
    for _ in range(max_new):
        context_ids = generation_context(ids, model.max_len)
        jamo = tokenizer.jamo_ids(context_ids.cpu()).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(context_ids.unsqueeze(0), jamo)[0, -1]
        logits[tokenizer.pad_id] = float("-inf")
        if repetition_penalty > 1.0:
            for token_id in ids.unique():
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        next_id = logits.argmax().view(1)
        ids = torch.cat([ids, next_id])
        if int(next_id) == tokenizer.stoi["\n"]:
            break
    return tokenizer.decode(ids[len(prompt_ids):].cpu()).rstrip("\n")


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
    model = FullSyllableCausalLM(
        tokenizer.get_vocab_size(), tokenizer.jamo.get_vocab_sizes(),
        checkpoint.get("emb_dim", 16), checkpoint.get("hidden_dim", 64),
        checkpoint.get("layers", 1), checkpoint.get("heads", 4),
        checkpoint.get("max_len", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = [json.loads(line) for line in (Path(args.data_dir) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    results = []
    for row in rows:
        prompt = f"Q: {row['question']}|"
        output = generate(model, tokenizer, prompt, device,
                          repetition_penalty=args.repetition_penalty)
        results.append({"question": row["question"], "expected": row["answer"],
                        "output": output, "exact": output == row["answer"],
                        "prefix10": output[:10] == row["answer"][:10]})
    report = {"rows": len(results), "exact": sum(r["exact"] for r in results),
              "prefix10": sum(r["prefix10"] for r in results), "results": results}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
