"""Create concise, balanced Korean dialogue/fact SFT examples."""

import argparse
import random
from pathlib import Path


def build_pool():
    pool = []
    greetings = (
        ("안녕!", "안녕하세요! 무엇을 도와드릴까요?"),
        ("좋은 아침이에요.", "좋은 아침이에요! 오늘도 잘 지내봐요."),
        ("처음 뵙겠습니다.", "저도 반갑습니다. 편하게 말씀해 주세요."),
        ("잘 지냈어?", "네, 잘 지냈어요. 당신은 어때요?"),
        ("반가워요.", "저도 반가워요!"),
    )
    for prompt, answer in greetings:
        for suffix in ("", " 오늘 이야기하고 싶어요.", " 잠깐 대화할까요?"):
            pool.append((prompt + suffix, answer))

    facts = {
        "우리나라의 수도를 알려줘": "서울입니다.",
        "한국의 수도는 어디야?": "서울입니다.",
        "물을 이루는 화학식은 무엇인가요?": "H2O입니다.",
        "한글을 만든 왕은 누구인가요?": "세종대왕입니다.",
        "고양이는 어떤 종류의 동물인가요?": "포유류인 동물입니다.",
        "파이썬은 어떤 도구인가요?": "프로그래밍 언어입니다.",
        "인공지능은 무엇인가요?": "배우고 판단하는 컴퓨터 기술입니다.",
        "지구의 위성은 무엇인가요?": "달입니다.",
    }
    for prompt, answer in facts.items():
        pool.append((prompt, answer))

    for a in range(1, 61):
        for b in range(1, 21):
            pool.extend([
                (f"{a} 더하기 {b}는 얼마야?", f"{a + b}입니다."),
                (f"{a}에 {b}를 더하면?", f"{a + b}입니다."),
            ])
            if a >= b:
                pool.append((f"{a}에서 {b}를 빼면 얼마인가요?", f"{a - b}입니다."))
            if a * b <= 120:
                pool.append((f"{a} 곱하기 {b}는?", f"{a * b}입니다."))

    cities = ("서울", "부산", "대구", "광주", "제주")
    foods = ("사과", "바나나", "빵", "김밥", "커피")
    names = ("민수", "지영", "수진", "현우", "서연")
    for name in names:
        for city in cities:
            pool.append((f"{name}는 {city}에 살고 있다. {name}가 사는 곳은", f"{city}입니다."))
        for food in foods:
            pool.append((f"{name}는 {food}를 샀다. {name}가 산 것은", f"{food}입니다."))

    definitions = (
        ("대화할 때 가장 중요한 태도는 무엇인가요?", "상대의 말을 잘 듣는 태도입니다."),
        ("모르는 것을 물어봐도 되나요?", "네, 모르는 것은 편하게 물어봐도 됩니다."),
        ("공부를 잘하려면 어떻게 해야 하나요?", "작은 목표를 세우고 꾸준히 연습하면 됩니다."),
        ("실수하면 어떻게 해야 하나요?", "원인을 살펴보고 다시 시도하면 됩니다."),
        ("오늘 할 일을 정리해줘.", "중요한 일부터 차례대로 정리해 보세요."),
    )
    pool.extend(definitions)
    return list(dict.fromkeys(pool))


def build_focused_pool():
    """Balanced dialogue pool with all arithmetic/number prompts removed."""
    pool = []
    base = build_pool()
    for prompt, answer in base:
        if any(word in prompt for word in ("더하기", "더하면", "더한", "곱하기", "곱", "빼면", "빼", "나누")):
            continue
        if any(char.isdigit() for char in prompt + answer):
            continue
        pool.append((prompt, answer))

    # Surface-form diversity is more useful than repeating one exact prompt.
    # Keep this pool short-answer and number-free so a small HRM can learn the
    # intent-to-answer mapping without arithmetic dominating the gradients.
    variants = {
        "서울입니다.": (
            "한국의 수도를 알려 주세요.", "대한민국 수도가 어디인지 궁금해요.",
            "우리나라의 수도는 어디인가요?", "수도로 정해진 도시는 어디예요?",
            "한국에서 가장 중심이 되는 수도를 말해줘.",
        ),
        "세종대왕입니다.": (
            "한글을 만든 임금은 누구인가요?", "훈민정음을 만든 왕을 알려줘.",
            "한글 창제와 관련된 왕은 누구예요?", "우리 글자를 만든 사람은 누구인가요?",
        ),
        "포유류인 동물입니다.": (
            "고양이는 어떤 동물에 속하나요?", "고양이의 동물 분류가 궁금해요.",
            "고양이는 포유동물인가요?", "고양이를 생물 분류로 보면 무엇인가요?",
        ),
        "프로그래밍 언어입니다.": (
            "파이썬은 어떤 언어인가요?", "파이썬의 종류를 설명해줘.",
            "파이썬으로 무엇을 할 수 있나요?", "파이썬은 컴퓨터 언어에 해당하나요?",
        ),
        "달입니다.": (
            "지구의 자연 위성 이름은 무엇인가요?", "지구를 도는 위성은 무엇이에요?",
            "밤하늘에서 지구 주위를 도는 천체를 알려줘.", "지구의 위성이 달이 맞나요?",
        ),
        "안녕하세요! 무엇을 도와드릴까요?": (
            "안녕하세요.", "안녕, 이야기하고 싶어요.", "오늘 대화할 수 있나요?",
            "도움을 받을 수 있을까요?", "처음 인사드려요.",
        ),
        "저도 반가워요!": (
            "만나서 반갑습니다.", "처음 뵙겠습니다.", "반가워요.",
            "만나서 기뻐요.",
        ),
        "원인을 살펴보고 다시 시도하면 됩니다.": (
            "실수했을 때는 어떻게 고치나요?", "잘못한 뒤에는 어떻게 해야 하나요?",
            "틀렸을 때 좋은 방법은 무엇인가요?", "실수하면 포기해야 하나요?",
        ),
        "작은 목표를 세우고 꾸준히 연습하면 됩니다.": (
            "공부 습관을 만들려면 어떻게 할까요?", "꾸준히 공부하는 비결이 있나요?",
            "공부를 계속하려면 무엇부터 해야 하나요?", "학습을 오래 지속하는 방법은요?",
        ),
        "중요한 일부터 차례대로 정리해 보세요.": (
            "오늘 할 일을 어떻게 정리할까요?", "해야 할 일이 많을 때는 어떻게 하나요?",
            "일의 우선순위를 정하는 방법을 알려줘.", "할 일을 정리하고 싶어요.",
        ),
    }
    for answer, prompts in variants.items():
        pool.extend((prompt, answer) for prompt in prompts)

    # Repeat short answer patterns: small models need multiple clean exposures
    # to learn answer-start and newline boundaries, not arithmetic memorization.
    focused = []
    for prompt, answer in pool:
        for repeat in range(12):
            suffix = "" if repeat == 0 else " (다시)"
            focused.append((prompt + suffix, answer))
    return focused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="train_data_hrm_dialogue_clean")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--focused-dialogue", action="store_true",
                        help="Exclude arithmetic and repeat clean dialogue patterns")
    args = parser.parse_args()
    pool = build_focused_pool() if args.focused_dialogue else build_pool()
    if args.count > len(pool):
        raise ValueError(f"requested {args.count}, but clean pool has only {len(pool)} examples")
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pool[:args.count]) + "\n"
    (output / "clean_dialogue_sft.txt").write_text(text, encoding="utf-8")
    print(f"generated={args.count} pool={len(pool)}")


if __name__ == "__main__":
    main()
