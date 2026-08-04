import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data/source/reasoning"


def text_of(row):
    q = row.get("question", "").strip()
    a = (row.get("answer") or row.get("solution") or row.get("reasoning") or "").strip()
    if not q or not a:
        return None
    return f"질문: {q}\n답변: {a}\n"


def read_records(path, seen, out, limit):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(out) >= limit:
                return
            try:
                text = text_of(json.loads(line))
            except (ValueError, TypeError):
                continue
            if not text:
                continue
            key = hashlib.sha1(text.encode()).digest()
            if key not in seen:
                seen.add(key)
                out.append(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    sources = (("instruction_normalized.jsonl", 0.40), ("korquad_normalized.jsonl", 0.30), ("owned_verified.jsonl", 0.20), ("gsm8k_ko_verified.jsonl", 0.10))
    for name, share in sources:
        read_records(SRC / name, seen, records, max(1, int(args.max_records * share)))
    cut = max(1, int(len(records) * 0.98))
    (args.out / "train.txt").write_text("\n".join(records[:cut]), encoding="utf-8")
    (args.out / "valid.txt").write_text("\n".join(records[cut:]), encoding="utf-8")
    raw = args.out / "raw_korquad.txt"
    if raw.exists():
        with (args.out / "train_expanded.txt").open("wb") as expanded:
            for source in (args.out / "train.txt", raw):
                with source.open("rb") as src:
                    shutil.copyfileobj(src, expanded)
    meta = {"records": len(records), "train_records": cut, "valid_records": len(records) - cut, "deduplicated": True, "expanded_corpus": str(args.out / "train_expanded.txt") if raw.exists() else None}
    (args.out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
