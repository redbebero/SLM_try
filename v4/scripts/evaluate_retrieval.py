"""Evaluate a character n-gram retrieval baseline on held-out SFT rows."""

import argparse
import json
from pathlib import Path

from retrieval import CharNgramRetriever, TfidfCharNgramRetriever


def load(path):
    return [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def char_overlap(left, right):
    left, right = set(left.replace(" ", "")), set(right.replace(" ", ""))
    return len(left & right) / max(1, len(left | right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--tfidf", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--category")
    args = parser.parse_args()
    retriever_cls = TfidfCharNgramRetriever if args.tfidf else CharNgramRetriever
    retriever = retriever_cls(load(args.train_dir))
    rows = load(args.test_dir)[:args.limit]
    exact = prefix = accepted = 0
    overlaps = []
    for row in rows:
        search_args = {"top_k": 1}
        if args.tfidf:
            search_args.update(source=args.source, category=args.category)
        hit = retriever.search(row["question"], **search_args)[0] if retriever.rows else {"answer": "", "score": 0.0}
        output = hit["answer"] if hit["score"] >= args.min_score else ""
        exact += output == row["answer"]
        prefix += output[:10] == row["answer"][:10]
        accepted += bool(output)
        overlaps.append(char_overlap(output, row["answer"]))
        print(json.dumps({"question": row["question"], "answer": row["answer"], "output": output, "score": hit["score"]}, ensure_ascii=False))
    total = max(1, len(rows))
    print(f"summary exact={exact}/{total} prefix10={prefix}/{total} accepted={accepted}/{total} answer_char_overlap={sum(overlaps)/total:.3f}")


if __name__ == "__main__":
    main()
