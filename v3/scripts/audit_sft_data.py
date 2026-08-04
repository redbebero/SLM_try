"""Clean Q/A SFT blocks without modifying the original dataset."""

import argparse
import hashlib
import json
import re
from pathlib import Path


REFUSAL_PATTERNS = (
    "아는 지식이 없어",
    "배경지식이 없어",
    "상식이 안 들어",
    "지문을 주시면",
    "정보가 담긴 지문",
)

QUOTED_FINAL_ANSWER = re.compile(
    r"질문의\s*답은\s*[\"“](.+?)[\"”]\s*입니다"
)
HASHED_FINAL_ANSWER = re.compile(r"(?:^|\n)\s*####\s*(.+?)\s*$", re.DOTALL)


def normalize_answer(question, answer):
    """Shorten only answers with an unambiguous, already-present final answer."""
    del question  # Reserved for future dataset-specific rules.
    quoted = QUOTED_FINAL_ANSWER.search(answer)
    if quoted:
        return f"{quoted.group(1).strip()}입니다."
    hashed = HASHED_FINAL_ANSWER.search(answer)
    if hashed:
        final = hashed.group(1).strip()
        if final:
            return f"정답은 {final}입니다."
    return answer


def audit_blocks(blocks):
    kept = []
    seen = set()
    stats = {
        "input": len(blocks),
        "kept": 0,
        "duplicate": 0,
        "malformed": 0,
        "low_quality": 0,
        "normalized": 0,
    }
    for block in blocks:
        block = re.sub(r"[ \t]+", " ", block.strip())
        parts = block.split("\nA: ")
        if len(parts) != 2 or not parts[0].startswith("Q: ") or not parts[0][3:].strip() or not parts[1].strip():
            stats["malformed"] += 1
            continue
        question, answer = parts[0][3:].strip(), parts[1].strip()
        if len(question) > 1000 or len(answer) > 2000 or "�" in block:
            stats["malformed"] += 1
            continue
        if any(pattern in answer for pattern in REFUSAL_PATTERNS):
            stats["low_quality"] += 1
            continue
        normalized_answer = normalize_answer(question, answer)
        if normalized_answer != answer:
            stats["normalized"] += 1
        normalized = (question.casefold(), normalized_answer.casefold())
        digest = hashlib.sha1("\0".join(normalized).encode("utf-8")).hexdigest()
        if digest in seen:
            stats["duplicate"] += 1
            continue
        seen.add(digest)
        kept.append(f"Q: {question}\nA: {normalized_answer}")
    stats["kept"] = len(kept)
    return kept, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = [block for block in source.read_text(encoding="utf-8").split("\n\n") if block.strip()]
    kept, stats = audit_blocks(blocks)
    (output_dir / "korquad_sft.txt").write_text("\n\n".join(kept) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
