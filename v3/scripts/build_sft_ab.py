"""Build a deterministic, non-destructive curated SFT subset for A/B tests."""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_sft_data import REFUSAL_PATTERNS


def build_curated_blocks(source, target_size=1000, seed=7):
    blocks = [b.strip() for b in Path(source).read_text(encoding="utf-8").split("\n\n") if b.strip()]
    eligible = []
    for block in blocks:
        parts = block.split("\nA: ", 1)
        if len(parts) != 2 or any(pattern in parts[1] for pattern in REFUSAL_PATTERNS):
            continue
        answer = parts[1]
        if "질문의 답은" in answer or "####" in answer:
            eligible.append(block)

    rng = random.Random(seed)
    remaining = [
        b for b in blocks
        if b not in eligible and "\nA: " in b
        and not any(pattern in b.split("\nA: ", 1)[1] for pattern in REFUSAL_PATTERNS)
    ]
    rng.shuffle(remaining)
    selected = eligible + remaining[: max(0, target_size - len(eligible))]
    rng.shuffle(selected)
    return selected[:target_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = build_curated_blocks(args.input, args.size)
    (output_dir / "korquad_sft.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print({"selected": len(blocks), "explicit_final": sum("질문의 답은" in b for b in blocks), "gsm8k_final": sum("####" in b for b in blocks)})


if __name__ == "__main__":
    main()
