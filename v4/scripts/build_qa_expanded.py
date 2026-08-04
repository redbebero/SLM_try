"""Create a larger paraphrase-varied Korean QA diagnostic set."""

from pathlib import Path


FACTS = [
    ("대한민국의 수도", "서울입니다."), ("1 더하기 1", "2입니다."),
    ("물이 어는 섭씨 온도", "0도입니다."), ("일주일의 날짜 수", "7일입니다."),
    ("한 시간의 분 수", "60분입니다."), ("삼각형의 변 수", "3개입니다."),
    ("지구의 위성", "달입니다."), ("태양의 종류", "항성입니다."),
    ("봄 다음 계절", "여름입니다."), ("겨울 다음 계절", "봄입니다."),
    ("해가 뜨는 방향", "동쪽입니다."), ("해가 지는 방향", "서쪽입니다."),
    ("호흡에 필요한 기체", "산소입니다."), ("식물의 양분 생성 과정", "광합성입니다."),
    ("10에서 3을 뺀 값", "7입니다."), ("4 곱하기 5의 값", "20입니다."),
    ("100을 4로 나눈 값", "25입니다."), ("피를 보내는 기관", "심장입니다."),
    ("비를 피하는 물건", "우산입니다."), ("책을 빌리는 장소", "도서관입니다."),
    ("1년의 개월 수", "12개월입니다."), ("물의 화학식", "H2O입니다."),
    ("글자를 입력하는 장치", "키보드입니다."), ("화면을 보여 주는 장치", "모니터입니다."),
    ("아픈 사람을 진료하는 직업", "의사입니다."), ("수업을 가르치는 사람", "선생님입니다."),
    ("나무가 모인 곳", "숲입니다."), ("얼음이 녹은 상태", "물입니다."),
    ("불을 끄는 장비", "소화기입니다."), ("길을 건널 때 보는 신호", "신호등입니다."),
]


TRAIN_TEMPLATES = ["{}은 무엇인가요?", "{}를 말해 주세요.", "{}의 정답은 무엇인가요?", "{}에 대한 답을 알려 주세요.", "{}을 질문하면 어떤 답을 해야 하나요?", "{}는 무엇인지 간단히 답하세요."]
VALID_TEMPLATES = ["{}에 해당하는 것을 한 가지로 답하면 무엇인가요?", "{}에 대해 정확한 답을 말해 주세요."]


def main():
    root = Path("experiments/qa_expanded")
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    train = [f"{template.format(subject)}\t{answer}" for subject, answer in FACTS for template in TRAIN_TEMPLATES]
    valid = [f"{template.format(subject)}\t{answer}" for subject, answer in FACTS for template in VALID_TEMPLATES]
    (root / "train/train.tsv").write_text("\n".join(train) + "\n", encoding="utf-8")
    (root / "valid/valid.tsv").write_text("\n".join(valid) + "\n", encoding="utf-8")
    print(f"train={len(train)} valid={len(valid)} facts={len(FACTS)}")


if __name__ == "__main__":
    main()
