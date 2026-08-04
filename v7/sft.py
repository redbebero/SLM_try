import argparse, json
from pathlib import Path
import sentencepiece as spm
import torch
from torch.nn.utils import clip_grad_norm_
from src.korean_lm.model import Config, KoreanLM


def examples(path, tok):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        messages = row["messages"]
        assert messages[-1]["role"] == "assistant"
        prefix = ""
        for message in messages[:-1]:
            prefix += f"<|{message['role']}|>\n{message['content']}\n"
        prefix += "<|assistant|>\n"
        full = prefix + messages[-1]["content"] + "\n<|end|>"
        ids = tok.encode(full, out_type=int)
        prompt_len = len(tok.encode(prefix, out_type=int))
        yield ids, prompt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/pretrain.pt")
    ap.add_argument("--chat", default="data/chat_seed.jsonl")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.model")
    ap.add_argument("--out", default="artifacts/chat.pt")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = KoreanLM(Config(**ckpt["config"])).to(device); model.load_state_dict(ckpt["model"])
    tok = spm.SentencePieceProcessor(model_file=args.tokenizer)
    data = list(examples(args.chat, tok))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    metrics = []
    for step in range(args.steps):
        ids, prompt_len = data[step % len(data)]
        if len(ids) > model.cfg.block_size:
            ids = ids[:model.cfg.block_size]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        mask_start = max(0, prompt_len - 1)
        y[:, :mask_start] = -100
        opt.zero_grad(set_to_none=True); _, loss = model(x, y)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "loss": round(loss.item(), 4)}; metrics.append(row); print(json.dumps(row))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.__dict__, "model": model.state_dict()}, args.out)
    Path(args.out + ".metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
