"""Build a clean Korean dialogue SFT split from downloaded KoCulture parquet."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from tokenizer import KoJamoTokenizer


def normalise(value, limit):
    value = re.sub(r"\s+", " ", str(value)).strip()
    if not value or "�" in value or len(value) > limit:
        return None
    return value


def supported(value, tokenizer):
    return all(
        ("가" <= char <= "힣")
        or char in tokenizer.sym_vocab
        or char in tokenizer.eng_vocab
        or char in tokenizer.num_vocab
        or char in tokenizer.standalone_cho
        or char in tokenizer.standalone_jung
        or char in tokenizer.standalone_jong
        for char in value
    )


def make_record(row, tokenizer, index):
    question = normalise(row["question"], 260)
    answer = normalise(row["answer"], 420)
    title = normalise(row["title"], 80)
    if not question or not answer or not title:
        return None
    if len(question) + len(answer) > 680:
        return None
    if not all(supported(value, tokenizer) for value in (question, answer)):
        return None
    pair_hash = hashlib.sha256(f"{question}\n{answer}".encode()).hexdigest()
    return {
        "question": question,
        "answer": answer,
        "source": "huggingface-KREW:KoCulture-Dialogues",
        "source_id": str(index),
        "category": "natural_dialogue_slang",
        "title": title,
        "pair_hash": pair_hash,
    }


def write_split(path, records):
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (path / "verified.txt").write_text(
        "\n\n".join(f"Q: {row['question']}\nA: {row['answer']}" for row in records) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    dataframe = pd.read_parquet(args.input)
    tokenizer = KoJamoTokenizer()
    records = []
    seen = set()
    for index, row in dataframe.iterrows():
        record = make_record(row, tokenizer, int(index))
        if record and record["pair_hash"] not in seen:
            records.append(record)
            seen.add(record["pair_hash"])

    # Keep all examples of a slang/topic title in one split.
    titles = sorted({row["title"] for row in records})
    split_by_title = {
        title: "test" if index % 10 == 0 else "valid" if index % 10 == 1 else "train"
        for index, title in enumerate(titles)
    }
    splits = {name: [] for name in ("train", "valid", "test")}
    for row in records:
        splits[split_by_title[row["title"]]].append(row)
    output = Path(args.output_dir)
    for name, rows in splits.items():
        write_split(output / name, rows)
    manifest = {
        "source": "huggingface-KREW/KoCulture-Dialogues",
        "license": "CC BY-NC-SA 4.0",
        "input_rows": len(dataframe),
        "kept_rows": len(records),
        "titles": len(titles),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "title_overlap": len(
            set(row["title"] for row in splits["train"])
            & set(row["title"] for row in splits["valid"])
        ) + len(
            set(row["title"] for row in splits["train"])
            & set(row["title"] for row in splits["test"])
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
