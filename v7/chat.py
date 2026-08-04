import argparse
import torch
import sentencepiece as spm
from src.korean_lm.model import Config, KoreanLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/chat.pt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.model")
    ap.add_argument("--prompt", default="안녕하세요.")
    ap.add_argument("--tokens", type=int, default=80)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = KoreanLM(Config(**ckpt["config"])); model.load_state_dict(ckpt["model"])
    tok = spm.SentencePieceProcessor(model_file=args.tokenizer)
    prompt = f"<|user|>\n{args.prompt}\n<|assistant|>\n"
    ids = torch.tensor([tok.encode(prompt, out_type=int)])
    out = model.generate(ids, args.tokens)[0].tolist()
    print(tok.decode(out).split("<|end|>", 1)[0].strip())


if __name__ == "__main__":
    main()
