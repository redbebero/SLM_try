"""Build a larger, filtered SFT set from data already present in the project."""

import argparse
import json
import random
import re
from pathlib import Path


REFUSAL_PATTERNS = (
    "아는 지식이 없어",
    "배경지식이 없어",
    "지문을 주시면",
    "정보가 담긴 지문",
    "지식이 들어있지 않습니다",
)
PAIR_RE = re.compile(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:\s*|\Z)", re.DOTALL)


def valid_pair(question, answer, max_question=220, max_answer=360, max_total=500):
    question = re.sub(r"\s+", " ", question).strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.sub(r"\*[^*]{0,300}\*", "", answer).strip()
    if not question or not answer:
        return None
    if len(question) > max_question or len(answer) > max_answer:
        return None
    if len(question) + len(answer) > max_total:
        return None
    if "�" in question + answer or any(p in answer for p in REFUSAL_PATTERNS):
        return None
    return f"Q: {question}\nA: {answer}"


def load_korquad(path, limit, rng):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = re.sub(r"\s+", " ", paragraph.get("context", "")).strip()
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer = answers[0].get("text", "")
                evidence = next(
                    (sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", context)
                     if answer and answer in sentence),
                    "",
                )
                if not evidence:
                    continue
                question = f"지문: {evidence} 질문: {qa.get('question', '')}"
                item = valid_pair(question, answer, max_question=420, max_answer=60, max_total=480)
                if item:
                    candidates.append(item)
    rng.shuffle(candidates)
    return candidates[:limit]


def load_pairs(path, limit, rng):
    text = Path(path).read_text(encoding="utf-8")
    candidates = []
    for question, answer in PAIR_RE.findall(text):
        item = valid_pair(question, answer)
        if item:
            candidates.append(item)
    # Keep only the first occurrence of each exact pair.
    candidates = list(dict.fromkeys(candidates))
    rng.shuffle(candidates)
    return candidates[:limit]


def load_gsm(path, limit, rng):
    return load_pairs(path, limit, rng)


def build(args):
    rng = random.Random(args.seed)
    groups = {
        "korquad": load_korquad(args.korquad, args.korquad_limit, rng),
        "gsm8k": load_gsm(args.gsm8k, args.gsm_limit, rng),
        "roleplay": load_pairs(args.roleplay, args.roleplay_limit, rng),
        "curated": load_pairs(args.curated, args.curated_limit, rng),
    }
    all_items = []
    seen = set()
    for name in ("korquad", "gsm8k", "roleplay", "curated"):
        for item in groups[name]:
            if item not in seen:
                seen.add(item)
                all_items.append(item)
    rng.shuffle(all_items)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "korquad_sft.txt").write_text("\n\n".join(all_items) + "\n", encoding="utf-8")
    stats = {name: len(items) for name, items in groups.items()}
    stats["total"] = len(all_items)
    (output / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--korquad", default="datasets/KorQuAD_v1.0_train.json")
    parser.add_argument("--gsm8k", default="datasets/gsm8k_ko.txt")
    parser.add_argument("--roleplay", default="datasets/roleplay_data.txt")
    parser.add_argument("--curated", default="train_data_sft_curated/korquad_sft.txt")
    parser.add_argument("--output-dir", default="train_data_sft_high_quality")
    parser.add_argument("--korquad-limit", type=int, default=5000)
    parser.add_argument("--gsm-limit", type=int, default=2000)
    parser.add_argument("--roleplay-limit", type=int, default=5000)
    parser.add_argument("--curated-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
