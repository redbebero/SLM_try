"""Stream a bounded clean subset from v6's raw KorQuAD JSONL."""
import argparse, html, json, re
from pathlib import Path

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean(value, limit=12000):
    value = re.sub(r"<(script|style)\b.*?</\1>", " ", value, flags=re.I | re.S)
    value = SPACE_RE.sub(" ", TAG_RE.sub(" ", html.unescape(value))).strip()
    return value[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/raw_korquad.txt"))
    ap.add_argument("--max-output-chars", type=int, default=180_000_000)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = records = 0
    with args.source.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if written >= args.max_output_chars:
                break
            try:
                row = json.loads(line)
                context = clean(row.get("context", ""))
                question = clean(row.get("question", ""), 2000)
                answer = clean((row.get("answer") or {}).get("text", ""), 1000)
            except (ValueError, TypeError, AttributeError):
                continue
            if not context or not question or not answer:
                continue
            text = f"문서: {context}\n질문: {question}\n답변: {answer}\n\n"
            if written + len(text) > args.max_output_chars:
                break
            dst.write(text)
            written += len(text); records += 1
    print(json.dumps({"records": records, "chars": written, "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
