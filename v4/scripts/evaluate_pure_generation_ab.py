"""Persist pure free-running comparison metrics for decoder experiments."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import generate, load_model
from tokenizer import KoJamoTokenizer


CASES = (
    "안녕하세요. 오늘 처음 대화하는데 괜찮으세요?",
    "괜히 마음이 답답하고 아무것도 하기 싫어.",
    "한국어 맞춤법을 오래 기억하려면 어떻게 공부해야 해?",
    "달은 지구 주변을 도는 위성 맞지?",
    "사과가 12개 있었는데 5개를 친구에게 주면 몇 개 남아?",
    "초성 ㅁ, 중성 ㅏ, 받침 ㄹ을 합치면 어떤 글자야?",
    "친구가 시험을 망쳤어. 짧고 자연스럽게 위로하는 말을 써줘.",
    "파이썬으로 숫자 두 개를 더하는 코드를 보여줘.",
    "이번 주 서울 날씨 알려줘.",
)


def metrics(output):
    compact = re.sub(r"\s+", "", output)
    grams = [compact[index:index + 3] for index in range(max(0, len(compact) - 2))]
    repeat = 1.0 - len(set(grams)) / len(grams) if grams else 0.0
    bad = sum(char in "?_" for char in output) / max(1, len(output))
    return {"repeat_3gram": repeat, "bad_char_ratio": bad}


def run(path, tokenizer, device):
    model, _ = load_model(path, tokenizer.get_vocab_sizes(), device=device)
    outputs = []
    for question in CASES:
        output = generate(
            model, tokenizer, f"Q: {question}\nA: ", max_new_chars=120,
            device=device, stop_on_newline=True, use_reasoning_router=False,
            allow_refusal=False,
        )
        outputs.append({"question": question, "output": output, "metrics": metrics(output)})
    values = [item["metrics"] for item in outputs]
    return {
        "mean_repeat_3gram": sum(item["repeat_3gram"] for item in values) / len(values),
        "max_repeat_3gram": max(item["repeat_3gram"] for item in values),
        "mean_bad_char_ratio": sum(item["bad_char_ratio"] for item in values) / len(values),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoJamoTokenizer()
    report = {name: run(path, tokenizer, device) for name, path in args.checkpoint}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in result.items() if key != "outputs"}
                     for name, result in report.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
