"""Build deterministic Korean arithmetic train/validation pairs."""

from pathlib import Path


def make(index, valid=False):
    a = (index * 37 + (11 if valid else 3)) % 91 + 2
    b = (index * 19 + (17 if valid else 5)) % 47 + 2
    c = (index * 13 + (7 if valid else 2)) % 19 + 2
    kind = index % 4
    if kind == 0:
        return f"연산 유형은 덧셈입니다. 연필이 {a}자루 있고 친구가 {b}자루를 더 주었습니다. 모두 몇 자루인가요?", f"{a + b}자루입니다."
    if kind == 1:
        a = max(a, b + 3)
        return f"연산 유형은 뺄셈입니다. 상자에 사과 {a}개가 있었는데 {b}개를 먹었습니다. 몇 개가 남았나요?", f"{a - b}개입니다."
    if kind == 2:
        return f"연산 유형은 곱셈입니다. 한 묶음에 {c}개씩 들어 있는 묶음이 {b}개 있습니다. 모두 몇 개인가요?", f"{c * b}개입니다."
    return f"연산 유형은 나눗셈입니다. {a}원을 {b}명이 똑같이 나누면 한 명이 받는 금액은 얼마인가요?", f"{a // b}원입니다."


def main():
    root = Path("experiments/arithmetic_qa")
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    for name, count, is_valid in (("train", 800, False), ("valid", 200, True)):
        rows = [make(i, is_valid) for i in range(count)]
        (root / name / f"{name}.tsv").write_text("\n".join(f"{q}\t{a}" for q, a in rows) + "\n", encoding="utf-8")
    print("train=800 valid=200")


if __name__ == "__main__":
    main()
