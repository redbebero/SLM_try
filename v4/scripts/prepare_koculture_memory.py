"""Convert the public KoCulture dialogue parquet into conservative Q/A memory."""

from pathlib import Path
import argparse
import re


def clean(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not 2 <= len(text) <= 220 or "http" in text.lower():
        return ""
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data_external/raw/KoCulture-Dialogues/data/train-00000-of-00001.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data_external/processed/koculture_sft.txt"))
    args = parser.parse_args()
    import pandas as pd
    frame = pd.read_parquet(args.input)
    blocks = []
    seen = set()
    for _, row in frame.iterrows():
        q, a = clean(row.get("question")), clean(row.get("answer"))
        if not q or not a or (q, a) in seen:
            continue
        seen.add((q, a))
        blocks.append(f"Q: {q}\nA: {a}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print({"rows": len(blocks), "license": "CC-BY-NC-SA-4.0"})


if __name__ == "__main__":
    main()
