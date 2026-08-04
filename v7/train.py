import argparse, json, math, random
from array import array
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
import sentencepiece as spm
from src.korean_lm.model import Config, KoreanLM, parameter_count


def encode(path, tokenizer, limit=None):
    ids = array("I")
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            ids.extend(tokenizer.encode(line, out_type=int))
            if limit and len(ids) >= limit:
                del ids[limit:]
                break
    return ids


def batches(ids, block, batch, device):
    usable = ((len(ids) - 1) // block) * block
    x = torch.tensor(ids[:usable], dtype=torch.long).view(-1, block)
    y = torch.tensor(ids[1:usable + 1], dtype=torch.long).view(-1, block)
    order = torch.randperm(len(x))
    for i in range(0, len(order), batch):
        j = order[i:i + batch]
        yield x[j].to(device), y[j].to(device)


@torch.no_grad()
def evaluate(model, ids, cfg, batch, device):
    vals = [loss.item() for x, y in batches(ids, cfg.block_size, batch, device) for _, loss in [model(x, y)]]
    return sum(vals) / len(vals) if vals else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.model")
    ap.add_argument("--train", default=None)
    ap.add_argument("--valid", default="data/processed/valid.txt")
    ap.add_argument("--out", default="artifacts/pretrain.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    args.train = args.train or ("data/processed/train_expanded.txt" if Path("data/processed/train_expanded.txt").exists() else "data/processed/train.txt")
    random.seed(0); torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    train_ids = encode(args.train, tokenizer, args.max_tokens)
    valid_ids = encode(args.valid, tokenizer, args.max_tokens // 5)
    cfg = Config(tokenizer.vocab_size(), args.block_size, args.layers, 8, args.hidden)
    model = KoreanLM(cfg).to(device)
    print(json.dumps({"device": device, "parameters": parameter_count(model), "train_tokens": len(train_ids)}))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    metrics = []
    best_val = float("inf")
    step = 0
    while step < args.steps:
        for x, y in batches(train_ids, cfg.block_size, args.batch_size, device):
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step == 1 or step % 100 == 0 or step == args.steps:
                val = evaluate(model, valid_ids, cfg, args.batch_size, device)
                row = {"step": step, "train_loss": round(loss.item(), 4), "valid_loss": round(val, 4), "perplexity": round(math.exp(min(val, 20)), 2)}
                metrics.append(row); print(json.dumps(row))
                if val < best_val:
                    best_val = val
                    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, str(args.out) + ".best.pt")
                model.train()
            if step >= args.steps: break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.out)
    Path(args.out + ".metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
