"""Measure partial answer accuracy and compare with a frequency baseline."""

import argparse
from collections import Counter

from core_generate import generate, load_model
from tokenizer import KoJamoTokenizer


def rows(path):
    for raw in open(path, encoding="utf-8"):
        text = raw.strip()
        if text and " " in text:
            prompt, answer = text.rsplit(" ", 1)
            yield prompt + " ", answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    args = parser.parse_args()
    train = list(rows(args.train))
    valid = list(rows(args.valid))
    common = Counter(answer for _, answer in train).most_common(1)[0][0]
    tokenizer = KoJamoTokenizer()
    model, device = load_model(args.checkpoint, tokenizer.get_vocab_sizes())
    exact = first = eos = 0
    char_correct = char_total = baseline_exact = 0
    outputs = Counter()
    length_stats = Counter()
    for prompt, answer in valid:
        predicted, ended = generate(model, tokenizer, prompt, len(answer), device, return_eos=True)
        outputs[predicted] += 1
        exact += predicted == answer
        eos += ended
        first += bool(predicted) and predicted[0] == answer[0]
        char_total += len(answer)
        char_correct += sum(a == b for a, b in zip(predicted, answer))
        baseline_exact += common == answer
        length_stats[len(answer)] += int(predicted == answer)
    n = len(valid)
    print(f"common_answer={common!r} common_exact={baseline_exact}/{n}")
    print(f"model_exact={exact}/{n} first_char={first}/{n} char_accuracy={char_correct}/{char_total} eos={eos}/{n}")
    print("exact_by_answer_length=", dict(sorted(length_stats.items())))
    print("top_outputs=", outputs.most_common(10))


if __name__ == "__main__":
    main()
