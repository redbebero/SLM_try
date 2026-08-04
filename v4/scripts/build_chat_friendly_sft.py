"""Build clean SFT records from Korean Chat Friendly CSV."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from tokenizer import KoJamoTokenizer


def clean(value, limit):
    value = re.sub(r"\s+", " ", str(value)).strip().rstrip(",")
    return value if value and "�" not in value and len(value) <= limit else None


def supported(value, tokenizer):
    return all(("가" <= c <= "힣") or c in tokenizer.sym_vocab or c in tokenizer.eng_vocab
               or c in tokenizer.num_vocab or c in tokenizer.standalone_cho
               or c in tokenizer.standalone_jung or c in tokenizer.standalone_jong for c in value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    tokenizer = KoJamoTokenizer()
    rows = []
    seen = set()
    for index, source in df.iterrows():
        question = clean(source["short_question"], 260)
        answer = clean(source["short_answer"], 420)
        if not question or not answer or len(question) + len(answer) > 680:
            continue
        if not supported(question, tokenizer) or not supported(answer, tokenizer):
            continue
        pair_hash = hashlib.sha256(f"{question}\n{answer}".encode()).hexdigest()
        if pair_hash in seen:
            continue
        seen.add(pair_hash)
        rows.append({"question": question, "answer": answer,
                     "source": "JaeJiMin:korean_chat_friendly", "source_id": str(index),
                     "category": "friendly_counseling", "pair_hash": pair_hash})
    rows.sort(key=lambda row: row["pair_hash"])
    n = len(rows)
    splits = {"test": rows[:n // 10], "valid": rows[n // 10:2 * n // 10], "train": rows[2 * n // 10:]}
    output = Path(args.output_dir)
    for name, values in splits.items():
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "records.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
        (target / "verified.txt").write_text("\n\n".join(f"Q: {v['question']}\nA: {v['answer']}" for v in values) + "\n", encoding="utf-8")
    manifest = {"source": "JaeJiMin/korean_chat_friendly", "license": "MIT", "input_rows": len(df),
                "kept_rows": len(rows), "splits": {k: len(v) for k, v in splits.items()}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
