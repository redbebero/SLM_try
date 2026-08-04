"""Build document-disjoint KorQuAD TSV splits."""

import json
import random
import re
from pathlib import Path


def collect(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = []
    for article in data["data"]:
        rows = []
        for paragraph in article["paragraphs"]:
            context = " ".join(paragraph["context"].split())
            for qa in paragraph["qas"]:
                if not qa.get("answers"):
                    continue
                question = " ".join(qa["question"].split())
                answer = " ".join(qa["answers"][0]["text"].split())
                evidence = next((sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", context) if answer in sentence), context[:180])
                prompt = f"지문: {evidence} 질문: {question}"
                if len(prompt) <= 220 and 0 < len(answer) <= 30:
                    rows.append((prompt, answer))
        if rows:
            articles.append(rows)
    return articles


def main():
    articles = collect("datasets/KorQuAD_v1.0_train.json")
    random.Random(42).shuffle(articles)
    cut = max(1, int(len(articles) * 0.8))
    train = [row for article in articles[:cut] for row in article][:500]
    valid = [row for article in articles[cut:] for row in article][:100]
    root = Path("experiments/korquad_qa")
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        (root / name / f"{name}.tsv").write_text("\n".join(f"{q}\t{a}" for q, a in rows) + "\n", encoding="utf-8")
    print(f"articles={len(articles)} train={len(train)} valid={len(valid)}")


if __name__ == "__main__":
    main()
