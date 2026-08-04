"""Build document-disjoint extractive KorQuAD span records."""

import json
import random
import re
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--train-limit", type=int, default=500); parser.add_argument("--valid-limit", type=int, default=100)
    args = parser.parse_args()
    raw = json.loads(Path("datasets/KorQuAD_v1.0_train.json").read_text(encoding="utf-8"))
    articles = []
    for article in raw["data"]:
        rows = []
        for paragraph in article["paragraphs"]:
            context = " ".join(paragraph["context"].split())
            for qa in paragraph["qas"]:
                if not qa.get("answers"):
                    continue
                answer = " ".join(qa["answers"][0]["text"].split())
                question = " ".join(qa["question"].split())
                sentence = next((s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if answer in s), "")
                if not sentence or len(sentence) > 180 or len(answer) > 30:
                    continue
                prompt = f"지문: {sentence} 질문: {question}"
                start = prompt.find(answer)
                if start < 0 or len(prompt) > 240:
                    continue
                rows.append({"prompt": prompt, "answer": answer, "start": start, "end": start + len(answer) - 1})
        if rows:
            articles.append(rows)
    random.Random(42).shuffle(articles)
    cut = max(1, int(len(articles) * 0.8))
    train = [row for article in articles[:cut] for row in article][:args.train_limit]
    valid = [row for article in articles[cut:] for row in article][:args.valid_limit]
    root = Path("experiments/korquad_span")
    (root / "train").mkdir(parents=True, exist_ok=True); (root / "valid").mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        (root / name / f"{name}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"articles={len(articles)} train={len(train)} valid={len(valid)}")


if __name__ == "__main__":
    main()
