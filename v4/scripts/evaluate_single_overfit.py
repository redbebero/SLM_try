"""Check free-running output against one real downloaded SFT row."""
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
    args = parser.parse_args()
    record = json.loads((Path(args.data_dir) / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    tokenizer = KoJamoTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, tokenizer.get_vocab_sizes(), device=device)
    prompt = f"Q: {record['question']}\nA: "
    output = generate(model, tokenizer, prompt, max_new_chars=120, device=device,
                      stop_on_newline=True, use_reasoning_router=False)
    expected = record["answer"]
    print(json.dumps({
        "prompt": prompt,
        "expected": expected,
        "output": output,
        "exact": output == expected,
        "prefix_10_exact": output[:10] == expected[:10],
        "output_len": len(output),
        "expected_len": len(expected),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
