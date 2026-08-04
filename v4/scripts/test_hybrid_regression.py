"""Regression gate for the small Korean-jamo HRM hybrid runtime."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import generate, load_model
from dialogue_intent import load_intent_checkpoint
from knowledge_memory import load_sft_memory
from tokenizer import KoJamoTokenizer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIALOGUE = os.path.join(ROOT, "checkpoints/hrm_context_copy_pure_dialogue_v2_12ep_best.pth")
INTENT = os.path.join(ROOT, "checkpoints/hrm_intent_pure_v3_best.pth")
REASONING = os.path.join(ROOT, "checkpoints/hrm_context_reasoning_order_finetune_2ep_best.pth")
MEMORY_FILES = [
    os.path.join(ROOT, "train_data_sft_high_quality/korquad_sft.txt"),
    os.path.join(ROOT, "train_data_sft_fact_dialog/korquad_sft.txt"),
    os.path.join(ROOT, "train_data_sft_filtered_high_quality/filtered_sft.txt"),
]


def main():
    tokenizer = KoJamoTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, device = load_model(DIALOGUE, tokenizer.get_vocab_sizes(), device=device)
    reasoning, _ = load_model(REASONING, tokenizer.get_vocab_sizes(), device=device)
    intent = load_intent_checkpoint(INTENT, device=device)
    available_memory = [path for path in MEMORY_FILES if os.path.exists(path)]
    # Derived SFT folders are optional in a clean checkout; the learned
    # dialogue and intent checkpoints remain independently testable.
    memory = load_sft_memory(available_memory)

    cases = (
        ("안녕, 오늘도 이야기할 수 있을까?", "도와"),
        ("오늘은 좀 우울한데", "힘드셨겠어요"),
        ("요즘 기분이 조금 가라앉아 있어.", "힘드셨겠어요"),
        ("직장에서 스트레스를 받아서 어떻게 해야할지 모르겠어요.", "스트레스를 받고"),
        ("요즘 운동을 안 해서 몸이 좀 무겁다.", "몸이 무겁고"),
        ("도움이 필요해", "도와드릴게요"),
        ("궁금한 걸 하나 물어봐도 될까", "편하게 말씀"),
        ("뵙게 되어 기쁩니다.", "반가"),
        ("틀린 일을 바로잡으려면 무엇을 해야 하나요?", "다시"),
        ("학습을 오래 이어 가는 요령이 있을까요?", "연습"),
        ("한국의 수도가 어디야?", "서울"),
        ("물의 화학식은", "H2O"),
        ("[자소] 초성 ㄱ, 중성 ㅏ, 종성 ㄴ을 합치면?", "간"),
        ("[산수] 3개가 있고 2개를 더한 뒤 1개를 빼면?", "4"),
        ("[순서] 봄 겨울 여름을 순서대로", "봄 여름 겨울"),
        ("현재 한국 대통령이 누구야?", "현재 확인할 수"),
        ("김치찌개 만드는 방법을 알려줘.", "현재 확인할 수"),
    )
    failures = []
    for prompt, expected in cases:
        output = generate(
            model, tokenizer, f"Q: {prompt}\nA: ", max_new_chars=100,
            device=device, stop_on_newline=True, memory=memory,
            intent_model=intent, reasoning_model=reasoning,
        )
        if expected not in output:
            failures.append((prompt, output, expected))
        print(f"{prompt} -> {output}")

    history = []
    for prompt in ("내 이름은 민수야", "안녕하세요", "내 이름이 뭐야?"):
        history.append(f"Q: {prompt}")
        full_prompt = "\n".join(history[-12:]) + "\nA: "
        output = generate(
            model, tokenizer, full_prompt, max_new_chars=100,
            device=device, stop_on_newline=True, intent_model=intent,
            reasoning_model=reasoning,
        )
        history.append(f"A: {output}")
        print(f"TURN {prompt} -> {output}")
    if "민수님이라고" not in history[-1]:
        failures.append(("multi-turn name recall", history[-1], "민수님이라고"))

    history = []
    for prompt in ("나는 주말마다 등산해", "내가 주말마다 뭘 한다고 했지?"):
        history.append(f"Q: {prompt}")
        full_prompt = "\n".join(history[-12:]) + "\nA: "
        output = generate(
            model, tokenizer, full_prompt, max_new_chars=100,
            device=device, stop_on_newline=True, intent_model=intent,
            reasoning_model=reasoning,
        )
        history.append(f"A: {output}")
        print(f"TURN {prompt} -> {output}")
    if "등산" not in history[-1]:
        failures.append(("multi-turn activity recall", history[-1], "등산"))

    if failures:
        print(f"FAILURES={len(failures)}")
        for failure in failures:
            print("FAIL", failure)
        raise SystemExit(1)
    print(f"PASS cases={len(cases) + 5}")


if __name__ == "__main__":
    main()
