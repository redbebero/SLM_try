import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if not match:
        raise ValueError("GSM8K answer has no #### final answer")
    return match.group(1)


def translate(texts, tokenizer, model, nllb=False):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.inference_mode():
        kwargs = {"max_new_tokens": 256, "num_beams": 3}
        if nllb:
            kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("kor_Hang")
        generated = model.generate(**encoded, **kwargs)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=108)
    parser.add_argument("--translator", default="facebook/nllb-200-distilled-600M")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for translation")

    source = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(args.limit))
    nllb = args.translator.startswith("facebook/nllb")
    tokenizer = AutoTokenizer.from_pretrained(args.translator, src_lang="eng_Latn" if nllb else None)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.translator).to("cuda")
    rows = []
    batch_size = 4
    for start in range(0, len(source), batch_size):
        batch = source[start : start + batch_size]
        questions = translate(batch["question"], tokenizer, model, nllb)
        solutions = translate([item.rsplit("####", 1)[0].strip() for item in batch["answer"]], tokenizer, model, nllb)
        for offset, (question, solution, original) in enumerate(zip(questions, solutions, batch["answer"])):
            answer = parse_gsm8k_answer(original)
            rows.append({
                "id": f"gsm8k-ko-{start + offset}",
                "category": "arithmetic",
                "question": question,
                "solution": solution,
                "answer": answer,
                "template_id": f"gsm8k-{start + offset}",
                "source": "openai/gsm8k",
                "translator": args.translator,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
