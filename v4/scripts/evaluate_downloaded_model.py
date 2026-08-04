"""Compare baseline and downloaded-data checkpoints with and without routing."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat import generate, load_model
from dialogue_intent import load_intent_checkpoint
from knowledge_memory import load_sft_memory
from tokenizer import KoJamoTokenizer


CASES = [
    {"id": "greeting", "question": "안녕하세요. 오늘 처음 대화하는데 괜찮으세요?", "kind": "dialogue"},
    {"id": "emotion", "question": "괜히 마음이 답답하고 아무것도 하기 싫어.", "kind": "dialogue"},
    {"id": "advice", "question": "한국어 맞춤법을 오래 기억하려면 어떻게 공부해야 해?", "kind": "dialogue"},
    {"id": "fact", "question": "달은 지구 주변을 도는 위성 맞지?", "kind": "fact"},
    {"id": "math", "question": "사과가 12개 있었는데 5개를 친구에게 주면 몇 개 남아?", "kind": "math"},
    {"id": "jamo", "question": "초성 ㅁ, 중성 ㅏ, 받침 ㄹ을 합치면 어떤 글자야?", "kind": "jamo"},
    {"id": "creative", "question": "친구가 시험을 망쳤어. 짧고 자연스럽게 위로하는 말을 써줘.", "kind": "creative"},
    {"id": "code", "question": "파이썬으로 숫자 두 개를 더하는 코드를 보여줘.", "kind": "code"},
    {"id": "unknown", "question": "이번 주 서울 날씨 알려줘.", "kind": "unknown"},
]


def load_bundle(checkpoint, tokenizer, device):
    model, _ = load_model(checkpoint, tokenizer.get_vocab_sizes(), device=device)
    reasoning, _ = load_model(
        "checkpoints/hrm_context_reasoning_order_finetune_2ep_best.pth",
        tokenizer.get_vocab_sizes(), device=device,
    )
    intent = load_intent_checkpoint("checkpoints/hrm_intent_pure_v3_best.pth", device=device)
    memory = load_sft_memory([])
    return model, reasoning, intent, memory


def run(checkpoint, tokenizer, device, routed):
    model, reasoning, intent, memory = load_bundle(checkpoint, tokenizer, device)
    rows = []
    for case in CASES:
        prompt = f"Q: {case['question']}\nA: "
        output = generate(
            model, tokenizer, prompt, max_new_chars=120, device=device,
            stop_on_newline=True, memory=memory if routed else None,
            intent_model=intent if routed else None,
            reasoning_model=reasoning if routed else None,
            use_reasoning_router=routed,
        )
        rows.append({**case, "output": output})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--downloaded", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoJamoTokenizer()
    report = {
        "baseline_routed": run(args.baseline, tokenizer, device, True),
        "downloaded_routed": run(args.downloaded, tokenizer, device, True),
        "baseline_pure": run(args.baseline, tokenizer, device, False),
        "downloaded_pure": run(args.downloaded, tokenizer, device, False),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
