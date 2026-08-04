"""Build balanced Q/A SFT files from audited local and downloaded sources."""

import csv
import hashlib
import json
import random
import re
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "verified_sft"


def clean(text, limit):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text or "�" in text or len(text) > limit:
        return None
    return text


def pair(question, answer):
    question, answer = clean(question, 260), clean(answer, 420)
    if not question or not answer or len(question) + len(answer) > 600:
        return None
    return f"Q: {question}\nA: {answer}"


def split_items(items, valid_ratio=0.1):
    train, valid = [], []
    for item in items:
        bucket = int(hashlib.sha256(item.encode("utf-8")).hexdigest()[:8], 16) % 1000
        (valid if bucket < int(valid_ratio * 1000) else train).append(item)
    return train, valid


def json_objects(path):
    """Read JSON objects even when a source row contains literal newlines."""
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


def load_quality(limit_per_source=10000):
    path = ROOT / "datasets/external/korean_quality_cleaned/train.jsonl"
    groups = {}
    for row in json_objects(path):
        source = row.get("source", "unknown")
        messages = row.get("messages", [])
        for index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            answer = next((item.get("content", "") for item in messages[index + 1:] if item.get("role") == "assistant"), "")
            item = pair(message.get("content", ""), answer)
            if item:
                groups.setdefault(source, []).append(item)
            break
    output = []
    for source, items in sorted(groups.items()):
        output.extend(list(dict.fromkeys(items))[:limit_per_source])
    return output


def load_education():
    output = []
    for path in (ROOT / "datasets/external/aihub_education_sample/train.jsonl",):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            messages = row.get("conversations", [])
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            answer = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            item = pair(f"[{row.get('category', '교육')}] {user}", answer)
            if item:
                output.append(item)
    return list(dict.fromkeys(output))


def load_chatbot():
    path = ROOT / "data_external/raw/ChatbotData.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return [item for row in csv.DictReader(stream) if (item := pair(row.get("Q", ""), row.get("A", "")))]


def load_empathetic():
    output = []
    path = ROOT / "data_external/raw/Empathetic_data.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        item = pair(row.get("instruction", ""), row.get("output", ""))
        if item:
            output.append(item)
    return list(dict.fromkeys(output))


def load_qa_text(path, limit):
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text)
    output = []
    for block in blocks:
        question = re.search(r"(?m)^Q:\s*(.+)$", block)
        answer = re.search(r"(?m)^A:\s*(.+)$", block)
        if question and answer:
            item = pair(question.group(1), answer.group(1))
            if item:
                output.append(item)
    return list(dict.fromkeys(output))[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-limit", type=int, default=0,
                        help="Maximum examples retained per source group; 0 keeps all.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.output_dir)
    random.seed(42)
    groups = {
        "quality": load_quality(),
        "education": load_education(),
        "chatbot": load_chatbot(),
        "empathetic": load_empathetic(),
        "persona": load_qa_text(ROOT / "datasets/persona_data.txt", 6000),
        "roleplay": load_qa_text(ROOT / "datasets/roleplay_data.txt", 6000),
    }
    if args.group_limit:
        groups = {
            name: random.sample(items, min(args.group_limit, len(items)))
            for name, items in groups.items()
        }
    train, valid = [], []
    stats = {}
    for name, items in groups.items():
        items = list(dict.fromkeys(items)); source_train, source_valid = split_items(items)
        train.extend(source_train); valid.extend(source_valid)
        stats[name] = {"total": len(items), "train": len(source_train), "valid": len(source_valid)}
    random.shuffle(train); random.shuffle(valid)
    for folder, rows in ((out / "train", train), (out / "valid", valid)):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "verified.txt").write_text("\n\n".join(rows) + "\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stats": stats, "train": len(train), "valid": len(valid)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
