import sys

sys.path.insert(0, "scripts")

from act_conditioned_selector import ActConditionedSelector, label_dialogue_act


def test_act_label_is_explicit_for_advice_and_emotion():
    assert label_dialogue_act("공부 방법을 추천해줘") == "질문_조언"
    assert label_dialogue_act("오늘 너무 우울하고 지쳤어") == "감정_표현"


def test_selector_prefers_same_dialogue_act():
    rows = [
        {"question": "공부 방법 알려줘", "answer": "작게 나누어 연습하세요.", "pair_hash": "a"},
        {"question": "오늘 우울해", "answer": "많이 힘들었겠어요.", "pair_hash": "b"},
    ]
    selector = ActConditionedSelector(rows)
    result = selector.select("공부 방법을 어떻게 알려줘?")
    assert result["dialogue_act"] == "질문_조언"
    assert result["answer"] == "작게 나누어 연습하세요."
