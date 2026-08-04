import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"[\s,\.。!?！？:;；]+", "", text)


def exact_match(output: str, answer: str) -> bool:
    output, answer = normalize(output), normalize(answer)
    if answer.isdigit():
        return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", output) is not None
    return answer in output


def configure_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(args.model))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    predictions = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        conversations = [[{"role": "user", "content": item["question"] + "\n풀이를 짧게 쓰고 마지막에 정답을 쓰세요."}] for item in batch]
        template_args = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt", "return_dict": True, "padding": True}
        if args.disable_thinking:
            template_args["enable_thinking"] = False
        inputs = tokenizer.apply_chat_template(conversations, **template_args)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prompt_length = inputs["input_ids"].shape[1]
        for item, output in zip(batch, generated[:, prompt_length:]):
            text = tokenizer.decode(output, skip_special_tokens=True).strip()
            predictions.append({**item, "prediction": text, "exact": exact_match(text, item["answer"]), "valid": bool(text)})

    total = len(predictions)
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "data": str(args.data),
        "count": total,
        "exact_accuracy": sum(item["exact"] for item in predictions) / total if total else 0.0,
        "format_validity": sum(item["valid"] for item in predictions) / total if total else 0.0,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("count", "exact_accuracy", "format_validity")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
