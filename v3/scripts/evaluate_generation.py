"""Evaluate fixed Korean generation cases across checkpoints."""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import generate, load_model
from knowledge_memory import load_sft_memory
from reasoning_router import try_reasoning_answer
from context_extractor import extract_passage_answer
from dialogue_intent import load_intent_checkpoint
from tokenizer import KoJamoTokenizer


def load_cases(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def repetition_3gram_ratio(text):
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    ngrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def score_generation(text, expected_keywords):
    def comparable(value):
        return re.sub(r"[^0-9A-Za-z가-힣]+", "", value.lower())

    normalized_text = comparable(text)
    keyword_hits = sum(
        1 for keyword in expected_keywords
        if comparable(keyword) and comparable(keyword) in normalized_text
    )
    hangul_chars = len(re.findall(r"[가-힣]", text))
    non_space_chars = len(re.sub(r"\s+", "", text))
    return {
        "keyword_hit_rate": keyword_hits / max(1, len(expected_keywords)),
        "repetition_3gram_ratio": repetition_3gram_ratio(text),
        "full_hangul_ratio": hangul_chars / max(1, non_space_chars),
        "new_tokens": len(text.split()),
    }


def evaluate_checkpoint(checkpoint, cases, device=None, use_reasoning_router=True,
                        sft_format=False, memory=None, intent_model=None):
    tokenizer = KoJamoTokenizer()
    model, device = load_model(checkpoint, tokenizer.get_vocab_sizes(), device=device)
    is_sft = sft_format or "sft" in Path(checkpoint).name.lower()
    rows = []
    for case in cases:
        prompt = f"Q: {case['prompt']}\nA: " if is_sft else case["prompt"]
        # Exact symbolic tasks always take precedence over lexical memory.
        routed = try_reasoning_answer(case["prompt"]) if use_reasoning_router else None
        retrieved = (memory.retrieve(case["prompt"]) if memory is not None and routed is None
                     else None)
        if routed is not None:
            output = routed
        elif retrieved is not None:
            output = retrieved["answer"]
        else:
            extracted = extract_passage_answer(case["prompt"])
            output = extracted if extracted is not None else generate(
                model, tokenizer, prompt, max_new_chars=case["max_new_chars"],
                device=device, stop_on_newline=is_sft,
                repetition_penalty=1.12,
                use_reasoning_router=use_reasoning_router,
                intent_model=intent_model,
            )
        metrics = score_generation(output, case["expected_keywords"])
        row = {"id": case["id"], "prompt": case["prompt"], "output": output, **metrics}
        if retrieved is not None:
            row["memory_score"] = retrieved["score"]
            row["source_question"] = retrieved["question"]
        rows.append(row)
    aggregate = {
        "checkpoint": checkpoint,
        "cases": len(rows),
        "keyword_hit_rate": sum(row["keyword_hit_rate"] for row in rows) / len(rows),
        "repetition_3gram_ratio": sum(row["repetition_3gram_ratio"] for row in rows) / len(rows),
        "full_hangul_ratio": sum(row["full_hangul_ratio"] for row in rows) / len(rows),
        "avg_new_tokens": sum(row["new_tokens"] for row in rows) / len(rows),
        "rows": rows,
    }
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cases", default="eval/ko_generation_cases.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-reasoning-router", action="store_true",
                        help="Measure the learned HRM only; do not use deterministic specialists.")
    parser.add_argument("--sft-format", action="store_true",
                        help="Evaluate with the Q/A prompt format used by SFT checkpoints.")
    parser.add_argument("--memory-files", nargs="*", default=None,
                        help="Optional SFT files used as lexical knowledge memory.")
    parser.add_argument("--memory-threshold", type=float, default=0.42)
    parser.add_argument("--intent-checkpoint", default=None)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    memory = None
    if args.memory_files:
        memory = load_sft_memory(args.memory_files)
        original_retrieve = memory.retrieve
        memory.retrieve = lambda query: original_retrieve(query, args.memory_threshold)
    intent_model = load_intent_checkpoint(args.intent_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu") \
        if args.intent_checkpoint else None
    results = [evaluate_checkpoint(
        checkpoint, cases, use_reasoning_router=not args.no_reasoning_router,
        sft_format=args.sft_format, memory=memory, intent_model=intent_model,
    ) for checkpoint in args.checkpoints]
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
