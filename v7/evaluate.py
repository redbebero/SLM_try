import argparse, json
from pathlib import Path
import torch
import sentencepiece as spm
from src.korean_lm.model import Config, KoreanLM

PROMPTS = [
    "안녕하세요.",
    "오늘 기분이 안 좋아.",
    "한국어로 자기소개해 줘.",
    "모르는 질문을 받으면 어떻게 해야 해?",
    "오늘 마음이 좀 가라앉지 않아.",
    "잠깐 쉬고 싶은데 뭘 하면 좋을까?",
    "내일 해야 할 일이 세 가지인데 정리가 안 돼.",
    "<|user|>\n오늘 기분이 안 좋아.\n<|assistant|>\n무슨 일이 있었는지 말해줘.\n<|user|>\n일이 너무 많았어.\n<|assistant|>\n",
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="artifacts/chat.pt"); ap.add_argument("--tokenizer", default="artifacts/tokenizer.model"); ap.add_argument("--tokens", type=int, default=50); ap.add_argument("--temperature", type=float, default=0.2); ap.add_argument("--out", default="artifacts/evaluation.jsonl"); args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu"); model = KoreanLM(Config(**ckpt["config"])); model.load_state_dict(ckpt["model"])
    tok = spm.SentencePieceProcessor(model_file=args.tokenizer); rows = []
    for prompt in PROMPTS:
        formatted = prompt if prompt.startswith("<|user|>") else f"<|user|>\n{prompt}\n<|assistant|>\n"
        ids = torch.tensor([tok.encode(formatted, out_type=int)])
        text = tok.decode(model.generate(ids, args.tokens, temperature=args.temperature, top_k=1)[0].tolist())
        text = text.split("<|end|>", 1)[0] + "<|end|>"
        rows.append({"prompt": prompt, "output": text})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(rows), "output": args.out}, ensure_ascii=False))


if __name__ == "__main__": main()
