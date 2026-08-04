"""Audit local Korean datasets before mixing them into training."""

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "dataset_audit" / "local_audit.json"


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_profile(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty = [line for line in lines if line.strip()]
    return {
        "format": "text",
        "rows": len(nonempty),
        "empty_rows": len(lines) - len(nonempty),
        "duplicate_rows": len(nonempty) - len(set(nonempty)),
    }


def csv_profile(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs = [norm(row.get("Q", "") + "\n" + row.get("A", "")) for row in rows]
    return {
        "format": "csv",
        "rows": len(rows),
        "missing_required": sum(not row.get("Q", "").strip() or not row.get("A", "").strip() for row in rows),
        "duplicate_pairs": len(pairs) - len(set(pairs)),
        "labels": sorted({row.get("label", "") for row in rows}),
    }


def jsonl_profile(path):
    rows = []
    invalid = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            invalid += 1
    fingerprints = [norm(json.dumps(row, ensure_ascii=False, sort_keys=True)) for row in rows]
    return {
        "format": "jsonl",
        "rows": len(rows),
        "invalid_rows": invalid,
        "duplicate_rows": len(fingerprints) - len(set(fingerprints)),
        "keys": sorted(set().union(*(row.keys() for row in rows if isinstance(row, dict)))) if rows else [],
    }


def korquad_profile(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    articles = data.get("data", [])
    qas = [qa for article in articles for paragraph in article.get("paragraphs", []) for qa in paragraph.get("qas", [])]
    answers = [qa.get("answers", []) for qa in qas]
    return {
        "format": "korquad_json",
        "articles": len(articles),
        "questions": len(qas),
        "questions_without_answers": sum(not answer for answer in answers),
        "answer_duplicates_within_question": sum(len({a.get("text", "") for a in answer}) != len(answer) for answer in answers),
    }


def profile(path):
    if path.name == "KorQuAD_v1.0_train.json":
        details = korquad_profile(path)
    elif path.suffix == ".csv":
        details = csv_profile(path)
    elif path.suffix == ".jsonl":
        details = jsonl_profile(path)
    else:
        details = text_profile(path)
    details.update({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    return details


def main():
    roots = [ROOT / "datasets", ROOT / "data_external" / "raw", ROOT / "train_data"]
    files = sorted(
        path for root in roots for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    )
    result = {"files": [profile(path) for path in files]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in result["files"]:
        print(item["path"], item.get("rows", item.get("questions", "-")), "dup=", item.get("duplicate_rows", item.get("duplicate_pairs", "-")))
    print("audit=", OUT)


if __name__ == "__main__":
    main()
