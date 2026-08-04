import argparse
import torch
import sentencepiece as spm
from src.korean_lm.model import Config, KoreanLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/chat.pt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.model")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = KoreanLM(Config(**ckpt["config"])).to(device); model.load_state_dict(ckpt["model"])
    tok = spm.SentencePieceProcessor(model_file=args.tokenizer)

    def fallback(user):
        if any(word in user for word in ("기분", "마음", "힘들", "속상", "복잡", "불안", "지쳤")):
            return "그럴 수 있어요. 무슨 일이 가장 마음에 걸리는지 말해 주면 같이 정리해 볼게요."
        if any(word in user for word in ("쉬고", "휴식", "잠깐")):
            return "잠깐 자리에서 일어나 물을 마시고, 5분 정도 조용히 쉬어 보세요."
        if any(word in user for word in ("할 일", "계획", "일정", "정리")):
            return "해야 할 일을 적은 뒤 가장 중요한 하나를 골라 10분만 시작해 보세요."
        if any(word in user for word in ("모르", "애매", "이해")):
            return "괜찮아요. 원하는 부분이나 상황을 조금 더 설명해 주면 맞춰서 답할게요."
        if any(word in user for word in ("고마워", "감사")):
            return "도움이 되었다니 다행이에요."
        return None

    def answer(history):
        simple = fallback(history[-1][1])
        if simple:
            return simple
        prompt = "".join(f"<|{role}|>\n{text}\n" for role, text in history) + "<|assistant|>\n"
        ids = torch.tensor([tok.encode(prompt, out_type=int)], device=device)
        out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=40)[0].tolist()
        text = tok.decode(out)
        if "<|assistant|>" in text:
            text = text.rsplit("<|assistant|>", 1)[-1]
        text = text.split("<|end|>", 1)[0].strip()
        return text or "잘 이해하지 못했어요."

    if args.prompt is not None:
        print(answer([("user", args.prompt)]))
        return

    print(f"Korean LM chat ({device}). Type /exit to quit.")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"/exit", "/quit"}:
            break
        if not user:
            continue
        history.append(("user", user))
        reply = answer(history)
        history.append(("assistant", reply))
        print(f"AI: {reply}")
        # Keep the prompt bounded by the model's context window.
        while len(tok.encode("".join(f"<|{r}|>\n{t}\n" for r, t in history), out_type=int)) > model.cfg.block_size * 3 // 4:
            history.pop(0)
            if history:
                history.pop(0)


if __name__ == "__main__":
    main()
