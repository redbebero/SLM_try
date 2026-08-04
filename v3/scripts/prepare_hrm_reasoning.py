"""Generate balanced, exact-answer Korean reasoning tasks for HRM-lite."""

import argparse
import random
import re
from pathlib import Path


KINDS = ("arithmetic", "jamo", "ordering")

CHOSEONG = ("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
            "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
JUNGSEONG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
JONGSEONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")


def _arithmetic_task(rng):
    a, b = rng.randint(2, 60), rng.randint(2, 40)
    c = rng.randint(1, min(25, a + b - 1))
    answer = a + b - c
    templates = (
        f"Q: [산수] 바구니에 {a}개가 있습니다. {b}개를 더 넣고 {c}개를 꺼냈습니다. 남은 개수는?",
        f"Q: [산수] 서랍에 연필이 {a}자루 있습니다. {b}자루를 넣은 뒤 {c}자루를 사용했습니다. 몇 자루가 남나요?",
        f"Q: [산수] 도서관에 책 {a}권이 있습니다. {b}권을 기증받고 {c}권을 빌려주었습니다. 남은 책은 몇 권인가요?",
        f"Q: [산수] 상점에 공 {a}개가 있습니다. {b}개를 배송받고 {c}개를 팔았습니다. 남은 공은 몇 개인가요?",
    )
    question = rng.choice(templates)
    return f"{question}\nA: {a}+{b}-{c}={answer}이므로 정답은 {answer}개입니다."


def _jamo_task(rng):
    cho_index = rng.randrange(len(CHOSEONG))
    jung_index = rng.randrange(len(JUNGSEONG))
    jong_index = rng.randrange(len(JONGSEONG))
    cho, jung, jong_symbol = CHOSEONG[cho_index], JUNGSEONG[jung_index], JONGSEONG[jong_index]
    answer = chr(0xAC00 + (cho_index * 21 + jung_index) * 28 + jong_index)
    jong = jong_symbol or "없음"
    if not jong_symbol:
        question = rng.choice((
            f"Q: [자소] 초성 {cho}과 중성 {jung}을 합치면 어떤 글자인가요?",
            f"Q: [자소] {cho} + {jung}을 조합하면 어떤 한글 음절이 되나요?",
            f"Q: [자소] {cho}와 {jung}으로 만들 수 있는 글자를 쓰세요.",
            f"Q: [자소] 초성은 {cho}, 중성은 {jung}입니다. 완성 글자는 무엇인가요?",
        ))
        explanation = f"{cho}과 {jung}을 합치면 {answer}"
    else:
        question = rng.choice((
            f"Q: [자소] 초성 {cho}, 중성 {jung}, 종성 {jong}을 합치면 어떤 글자인가요?",
            f"Q: [자소] {cho} + {jung} + 받침 {jong}을 조합하면 어떤 한글 음절이 되나요?",
            f"Q: [자소] {cho}, {jung}, 받침 {jong}으로 만들 수 있는 글자를 쓰세요.",
            f"Q: [자소] 초성 {cho}, 중성 {jung}, 종성 {jong}입니다. 완성 글자는 무엇인가요?",
        ))
        explanation = f"{cho}, {jung}, {jong}을 합치면 {answer}"
    return f"{question}\nA: {explanation}이므로 정답은 {answer}입니다."


def _ordering_task(rng):
    ordered_sets = (
        ("계절", ("봄", "여름", "가을", "겨울")),
        ("달", tuple(f"{number}월" for number in range(1, 13))),
        ("요일", ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")),
        ("숫자", tuple(str(number) for number in range(1, 31))),
    )
    category, sequence = rng.choice(ordered_sets)
    items = rng.sample(sequence, 3)
    order = sorted(items, key=sequence.index)
    question = rng.choice((
        f"Q: [순서] 다음 {category}을(를) 빠른 순서대로 배열하세요: {' '.join(items)}",
        f"Q: [순서] 다음 {category} 중 먼저 오는 것부터 나열하세요: {'·'.join(items)}",
        f"Q: [순서] {category}의 흐름에 맞게 정렬하세요: {', '.join(items)}",
        f"Q: [순서] {category}을(를) 시간 순서로 정리하세요: {' / '.join(items)}",
    ))
    return f"{question}\nA: {category}의 순서는 {' '.join(order)}입니다."


def task_type(task):
    """Return the explicit reasoning type embedded in a generated task."""
    if task.startswith("Q: [산수]"):
        return "arithmetic"
    if task.startswith("Q: [자소]"):
        return "jamo"
    if task.startswith("Q: [순서]"):
        return "ordering"
    raise ValueError(f"unknown HRM task type: {task[:40]!r}")


def compact_task(task):
    """Reduce a generated explanation to a short canonical answer."""
    prompt, answer = task.split("\nA: ", 1)
    kind = task_type(task)
    if kind == "arithmetic":
        numbers = [int(value) for value in re.findall(r"\d+", prompt)]
        return f"{prompt}\nA: {numbers[0] + numbers[1] - numbers[2]}"
    if kind == "jamo":
        match = re.search(r"정답은\s+(.+?)입니다", answer)
        return f"{prompt}\nA: {match.group(1).strip()}" if match else task
    match = re.search(r"순서는\s+(.+?)입니다", answer)
    return f"{prompt}\nA: {match.group(1).strip()}" if match else task


def canonical_task(task):
    """Keep only the structured fields needed by the reasoning operation."""
    prompt, answer = task.split("\nA: ", 1)
    kind = task_type(task)
    if kind == "arithmetic":
        numbers = [int(value) for value in re.findall(r"\d+", prompt)]
        return f"Q: [산수] A={numbers[0]} B={numbers[1]} C={numbers[2]}\nA: {numbers[0] + numbers[1] - numbers[2]}"
    if kind == "jamo":
        cho = re.search(r"초성(?:은)?\s*([ㄱ-ㅎ])", prompt)
        jung = re.search(r"중성(?:은)?\s*([ㅏ-ㅣ])", prompt)
        jong = re.search(r"(?:종성|받침)(?:은)?\s*([ㄱ-ㅎ])", prompt)
        if not cho or not jung:
            standalone = re.findall(r"[ㄱ-ㅎㅏ-ㅣ]", prompt)
            if len(standalone) >= 2:
                cho = cho or re.match(r".", standalone[0])
                jung = jung or re.match(r".", standalone[1])
                if len(standalone) >= 3:
                    jong = jong or re.match(r".", standalone[2])
        if not cho or not jung:
            return task
        cho_value = cho.group(1) if cho.lastindex else cho.group(0)
        jung_value = jung.group(1) if jung.lastindex else jung.group(0)
        final = (jong.group(1) if jong.lastindex else jong.group(0)) if jong else "없음"
        answer_match = re.search(r"정답은\s+(.+?)입니다", answer)
        answer_value = answer_match.group(1).strip() if answer_match else answer.strip()
        return f"Q: [자소] C={cho_value} V={jung_value} F={final}\nA: {answer_value}"
    sequence = {
        "계절": ("봄", "여름", "가을", "겨울"),
        "달": tuple(f"{n}월" for n in range(1, 13)),
        "요일": ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"),
        "숫자": tuple(str(n) for n in range(1, 31)),
    }
    category = next((name for name in sequence if name in prompt), "숫자")
    if category == "숫자":
        items = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", prompt)
    elif category == "달":
        items = re.findall(r"(?:[1-9]|1[0-2])월", prompt)
    else:
        items = [item for item in sequence[category] if item in prompt]
    items = list(dict.fromkeys(items))
    ordered = re.search(r"순서는\s+(.+?)입니다", answer)
    expected = ordered.group(1).strip() if ordered else " ".join(sorted(items, key=sequence[category].index))
    return f"Q: [순서] S={category} X={'|'.join(items)}\nA: {expected}"


def make_tasks(count, seed=7, compact=False, canonical=False):
    """Create deterministic, balanced, duplicate-free reasoning examples."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(seed)
    quotas = {kind: count // len(KINDS) for kind in KINDS}
    for kind in KINDS[:count % len(KINDS)]:
        quotas[kind] += 1
    builders = {
        "arithmetic": _arithmetic_task,
        "jamo": _jamo_task,
        "ordering": _ordering_task,
    }
    tasks = []
    for kind in KINDS:
        unique = set()
        while len(unique) < quotas[kind]:
            unique.add(builders[kind](rng))
        tasks.extend(unique)
    rng.shuffle(tasks)
    if canonical:
        return [canonical_task(task) for task in tasks]
    return [compact_task(task) for task in tasks] if compact else tasks


def make_tasks_for_type(count, kind, seed=7, compact=False, canonical=False):
    """Create duplicate-free tasks for one specialist head experiment."""
    if kind not in KINDS:
        raise ValueError(f"unknown HRM task type: {kind}")
    rng = random.Random(seed)
    builder = {
        "arithmetic": _arithmetic_task,
        "jamo": _jamo_task,
        "ordering": _ordering_task,
    }[kind]
    unique = set()
    while len(unique) < count:
        unique.add(builder(rng))
    tasks = list(unique)
    rng.shuffle(tasks)
    if canonical:
        return [canonical_task(task) for task in tasks]
    return [compact_task(task) for task in tasks] if compact else tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="train_data_hrm_reasoning")
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--kind", choices=KINDS, default=None)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.kind:
        tasks = make_tasks_for_type(
            args.count, args.kind, seed=args.seed,
            compact=args.compact, canonical=args.canonical,
        )
    else:
        tasks = make_tasks(args.count, seed=args.seed, compact=args.compact, canonical=args.canonical)
    (output / "reasoning_sft.txt").write_text("\n\n".join(tasks) + "\n", encoding="utf-8")
    print(f"generated={len(tasks)}")


if __name__ == "__main__":
    main()
