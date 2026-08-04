"""Normalize manually written natural holdout answers to training format."""

from pathlib import Path


def main():
    allowed = {"서울", "봄", "얼음", "서쪽", "산소", "우산", "놀이터", "광합성", "심장", "소화기", "신호등", "도서관", "겨울", "가을", "산소", "태양", "물", "새벽", "낮", "의사", "선생님", "숲"}
    rows = []
    for line in Path("experiments/natural_holdout_ko/valid/valid.txt").read_text(encoding="utf-8").splitlines():
        if " " not in line:
            continue
        question, answer = line.rsplit(" ", 1)
        if answer in allowed:
            rows.append((question, answer + "입니다."))
    root = Path("experiments/fair_natural_holdout_ko/valid")
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("valid.tsv").write_text("\n".join(f"{q}\t{a}" for q, a in rows) + "\n", encoding="utf-8")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
