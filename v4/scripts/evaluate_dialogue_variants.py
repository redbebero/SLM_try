"""Evaluate fixed-format and paraphrase generalization of the Korean runtime."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import generate, load_model
from dialogue_intent import load_intent_checkpoint
from knowledge_memory import load_sft_memory
from reasoning_router import try_reasoning_answer
from tokenizer import KoJamoTokenizer


CASES = [
    ("fact_q", "Q: 한국의 수도가 어디야?\nA: ", "서울"),
    ("fact_paraphrase", "Q: 우리나라에서 가장 중심이 되는 도시는 어디인가요?\nA: ", "서울"),
    ("formula", "Q: 물을 화학식으로 어떻게 쓰나요?\nA: ", "H2O"),
    ("math_seen", "Q: [산수] 3개가 있고 2개를 더한 뒤 1개를 빼면?\nA: ", "4"),
    ("math_new", "Q: [산수] 귤이 9개 있었는데 4개를 먹었어. 몇 개 남았어?\nA: ", "5"),
    ("jamo_seen", "Q: [자소] 초성 ㄱ, 중성 ㅏ, 종성 ㄴ을 합치면?\nA: ", "간"),
    ("jamo_new", "Q: 초성은 ㄴ, 중성은 ㅏ, 받침은 ㄱ인 글자는?\nA: ", "낙"),
    ("order_new", "Q: 가을, 봄, 겨울을 계절이 시작되는 순서로 정리해줘.\nA: ", "봄 가을 겨울"),
    ("emotion_seen", "Q: 오늘은 좀 우울한데\nA: ", "힘드셨"),
    ("emotion_new", "Q: 오늘 하루 종일 마음이 무겁고 지쳤어.\nA: ", "힘"),
    ("creative", "Q: 친구에게 보낼 짧은 응원 문장을 만들어줘.\nA: ", "__not_refusal__"),
    ("unknown", "Q: 현재 한국 대통령이 누구야?\nA: ", "현재 확인"),
]


def load_runtime(checkpoint, device):
    tokenizer = KoJamoTokenizer()
    model, _ = load_model(checkpoint, tokenizer.get_vocab_sizes(), device=device)
    reasoning, _ = load_model("checkpoints/hrm_context_reasoning_order_finetune_2ep_best.pth", tokenizer.get_vocab_sizes(), device=device)
    intent = load_intent_checkpoint("checkpoints/hrm_intent_pure_v3_best.pth", device=device)
    paths = [path for path in (
        "train_data_sft_high_quality/korquad_sft.txt",
        "train_data_sft_fact_dialog/korquad_sft.txt",
        "train_data_sft_filtered_high_quality/filtered_sft.txt",
    ) if os.path.exists(path)]
    return tokenizer, model, reasoning, intent, load_sft_memory(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, reasoning, intent, memory = load_runtime(args.checkpoint, device)
    hits = 0
    for label, prompt, expected in CASES:
        output = generate(model, tokenizer, prompt, max_new_chars=100, device=device,
                          stop_on_newline=True, memory=memory,
                          intent_model=intent, reasoning_model=reasoning)
        hit = ("현재 확인할 수 있는 정보가 없습니다." not in output) if expected == "__not_refusal__" else expected in output
        hits += int(hit)
        print(f"{label}\t{'PASS' if hit else 'FAIL'}\tinput={prompt!r}\toutput={output!r}")
    print(f"SUMMARY={hits}/{len(CASES)}")


if __name__ == "__main__":
    main()
