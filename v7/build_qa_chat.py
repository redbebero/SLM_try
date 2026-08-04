"""Convert the real v6 QA/reasoning records into chat-format SFT data."""
import argparse, hashlib, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/source/reasoning"))
    ap.add_argument("--out", type=Path, default=Path("data/chat_real.jsonl"))
    args = ap.parse_args()
    files = ("instruction_normalized.jsonl", "owned_verified.jsonl", "korquad_normalized.jsonl", "gsm8k_ko_verified.jsonl")
    seen, rows = set(), []
    for name in files:
        with (args.source / name).open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    question = str(row.get("question", "")).strip()
                    answer = str(row.get("answer") or row.get("solution") or row.get("reasoning") or "").strip()
                except (ValueError, TypeError):
                    continue
                if not question or not answer:
                    continue
                key = hashlib.sha1((question + "\n" + answer).encode()).digest()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"id": f"qa-{len(rows):06d}", "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
