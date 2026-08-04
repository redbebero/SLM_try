import argparse, json, random
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


def batch(data, size, block_size, device):
    selected = random.sample(data, min(size, len(data)))
    selected = [item for item in selected if len(item[0]) > 1]
    width = min(block_size, max(len(ids) - 1 for ids, _ in selected))
    xs, ys = [], []
    for ids, prompt_len in selected:
        ids = ids[:width + 1]
        x = ids[:-1] + [0] * (width - len(ids) + 1)
        y = ids[1:] + [-100] * (width - len(ids) + 1)
        y[:max(0, min(prompt_len - 1, width))] = [-100] * max(0, min(prompt_len - 1, width))
        xs.append(x[:width]); ys.append(y[:width])
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/pretrain.pt")
    ap.add_argument("--chat", default="data/chat_seed.jsonl")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.model")
    ap.add_argument("--out", default="artifacts/chat.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = KoreanLM(Config(**ckpt["config"])).to(device); model.load_state_dict(ckpt["model"])
    tok = spm.SentencePieceProcessor(model_file=args.tokenizer)
    data = [example for path in args.chat.split(",") for example in examples(path, tok)]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    metrics = []
    for step in range(args.steps):
        x, y = batch(data, args.batch_size, model.cfg.block_size, device)
        opt.zero_grad(set_to_none=True); _, loss = model(x, y)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "loss": round(loss.item(), 4)}; metrics.append(row); print(json.dumps(row))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.__dict__, "model": model.state_dict()}, args.out)
    Path(args.out + ".metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
