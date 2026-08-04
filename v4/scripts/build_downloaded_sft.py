"""Build SFT splits from downloaded, licensed source datasets only."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import sys
sys.path.insert(0, str(ROOT) if 'ROOT' in globals() else str(Path(__file__).resolve().parent))
from tokenizer import KoJamoTokenizer


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "datasets/external/korean_quality_cleaned/train.jsonl"
AIHUB_TRAIN = ROOT / "datasets/external/aihub_education_sample/train.jsonl"
AIHUB_TEST = ROOT / "datasets/external/aihub_education_sample/validation.jsonl"
TOKENIZER = KoJamoTokenizer()


def supported_text(text):
    """Keep text exactly representable by current compact tokenizer."""
    for char in text:
        if ("가" <= char <= "힣"
                or char in TOKENIZER.sym_vocab
                or char in TOKENIZER.eng_vocab
                or char in TOKENIZER.num_vocab
                or char in TOKENIZER.standalone_cho
                or char in TOKENIZER.standalone_jung
                or char in TOKENIZER.standalone_jong):
            continue
        return False
    return True


def json_objects(path):
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        row, position = decoder.raw_decode(text, position)
        yield row


def clean(value, limit):
    value = re.sub(r"\s+", " ", str(value)).strip()
    if not value or "�" in value or len(value) > limit:
        return None
    return value


def make_record(question, answer, source, source_id, category, raw):
    question = clean(question, 260)
    answer = clean(answer, 420)
    if not question or not answer or len(question) + len(answer) > 680:
        return None
    if not supported_text(question) or not supported_text(answer):
        return None
    raw_hash = hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    pair_hash = hashlib.sha256(f"{question}\n{answer}".encode("utf-8")).hexdigest()
    return {
        "question": question,
        "answer": answer,
        "source": source,
        "source_id": str(source_id),
        "category": category or "unknown",
        "raw_hash": raw_hash,
        "pair_hash": pair_hash,
    }


def first_pair(messages):
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        answer = next(
            (item.get("content", "") for item in messages[index + 1:]
             if item.get("role") == "assistant"),
            "",
        )
        return message.get("content", ""), answer
    return "", ""


def load_quality():
    records = []
    for index, row in enumerate(json_objects(QUALITY)):
        question, answer = first_pair(row.get("messages", []))
        record = make_record(
            question, answer, f"quality:{row.get('source', 'unknown')}",
            index, "dialogue", row,
        )
        if record:
            records.append(record)
    return records


def load_aihub(path, split):
    records = []
    for index, row in enumerate(json_objects(path)):
        question, answer = first_pair(row.get("conversations", []))
        source = f"aihub:{row.get('source', 'unknown')}:{split}"
        source_id = row.get("id", row.get("metadata", {}).get("original_id", index))
        record = make_record(
            question, answer, source, source_id,
            row.get("category", "education"), row,
        )
        if record:
            records.append(record)
    return records


def deduplicate(records):
    seen = set()
    output = []
    for record in records:
        if record["pair_hash"] in seen:
            continue
        seen.add(record["pair_hash"])
        output.append(record)
    return output


def split_quality(records, valid_ratio=0.1):
    train, valid = [], []
    threshold = int(valid_ratio * 1000)
    for record in records:
        bucket = int(record["raw_hash"][:8], 16) % 1000
        (valid if bucket < threshold else train).append(record)
    return train, valid


def write_split(folder, records):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "verified.txt").write_text(
        "\n\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in records) + "\n",
        encoding="utf-8",
    )
    (folder / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def build_dataset(max_question_chars=None, max_answer_chars=None):
    quality = deduplicate(load_quality())
    education_train = deduplicate(load_aihub(AIHUB_TRAIN, "train"))
    education_test = deduplicate(load_aihub(AIHUB_TEST, "validation"))
    def keep(record):
        return ((max_question_chars is None or len(record["question"]) <= max_question_chars)
                and (max_answer_chars is None or len(record["answer"]) <= max_answer_chars))
    quality = [record for record in quality if keep(record)]
    education_train = [record for record in education_train if keep(record)]
    education_test = [record for record in education_test if keep(record)]
    quality_train, quality_valid = split_quality(quality)
    train = deduplicate(quality_train + education_train)
    valid = deduplicate(quality_valid)
    test = education_test

    test_hashes = {record["pair_hash"] for record in test}
    train = [record for record in train if record["pair_hash"] not in test_hashes]
    valid = [record for record in valid if record["pair_hash"] not in test_hashes]
    return train, valid, test, {
        "raw_quality": len(list(json_objects(QUALITY))),
        "raw_aihub_train": len(list(json_objects(AIHUB_TRAIN))),
        "raw_aihub_validation": len(list(json_objects(AIHUB_TEST))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/downloaded_sft"))
    parser.add_argument("--smoke-limit", type=int, default=500)
    parser.add_argument("--max-question-chars", type=int, default=None)
    parser.add_argument("--max-answer-chars", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.output_dir)

    train, valid, test, raw_counts = build_dataset(
        args.max_question_chars, args.max_answer_chars
    )

    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        write_split(out / name, rows)
    random.seed(42)
    smoke_train = random.sample(train, min(args.smoke_limit, len(train)))
    smoke_valid = random.sample(valid, min(max(1, args.smoke_limit // 10), len(valid)))
    write_split(out / "smoke_train", smoke_train)
    write_split(out / "smoke_valid", smoke_valid)

    manifest = {
        "sources": [
            {"path": str(QUALITY.relative_to(ROOT)), "license": "CC BY-NC-SA 4.0; inherited from KULLM-v2 and KoAlpaca", "commercial_use": "restricted"},
            {"path": str(AIHUB_TRAIN.relative_to(ROOT)), "license": "CC BY-NC-SA 4.0; AI Hub terms apply"},
            {"path": str(AIHUB_TEST.relative_to(ROOT)), "license": "CC BY-NC-SA 4.0; evaluation only"},
        ],
        "splits": {name: len(rows) for name, rows in (("train", train), ("valid", valid), ("test", test))},
        "smoke_splits": {"train": len(smoke_train), "valid": len(smoke_valid)},
        "removed_invalid_or_duplicate": {
            "quality": raw_counts["raw_quality"] - len(train) - len(valid),
            "aihub_train": raw_counts["raw_aihub_train"] - sum(row["source"].startswith("aihub:") for row in train),
            "aihub_validation": raw_counts["raw_aihub_validation"] - len(test),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
