"""Train a compact SentencePiece subword causal SFT model."""

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from subword_lm import CompactSubwordCausalLM


class SubwordTokenizer:
    def __init__(self, model_path):
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = self.processor.pad_id()
        self.answer_id = self.processor.piece_to_id("<answer>")
        self.eos_id = self.processor.piece_to_id("<eos>")

    def encode(self, text):
        return torch.tensor(self.processor.encode(text, out_type=int), dtype=torch.long)

    def decode(self, ids):
        control = {self.pad_id, self.answer_id, self.eos_id}
        return self.processor.decode([int(value) for value in ids if int(value) not in control])

    def vocab_size(self):
        return self.processor.get_piece_size()


def train_tokenizer(data_dir, model_prefix, vocab_size):
    corpus = Path(model_prefix).with_suffix(".txt")
    rows = [json.loads(line) for line in (Path(data_dir) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("\n".join(f"{row['question']}\n{row['answer']}" for row in rows) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(model_prefix), vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0, normalization_rule_name="identity",
        user_defined_symbols="<answer>,<eos>", pad_id=0, unk_id=1, bos_id=-1,
        eos_id=-1, hard_vocab_limit=False,
    )


def load_records(path):
    return [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def source_balanced_weights(rows):
    counts = {}
    for row in rows:
        source = row.get("source", "")
        counts[source] = counts.get(source, 0) + 1
    return [1.0 / counts.get(row.get("source", ""), 1) for row in rows]


def loss_weights(targets, mask, eos_id, eos_weight):
    weights = mask.clone()
    if eos_weight != 1.0:
        weights = weights * torch.where(targets == eos_id, eos_weight, 1.0)
    return weights


def scheduled_inputs(inputs, predictions, answer_mask, probability):
    if probability <= 0:
        return inputs
    replace = (torch.rand_like(answer_mask[:, :-1]) < probability) & (answer_mask[:, :-1] > 0)
    mixed = inputs.clone()
    mixed[:, 1:] = torch.where(replace, predictions[:, :-1], mixed[:, 1:])
    return mixed


def encode_record(row, tokenizer):
    prompt = f"Q: {row['question']}<answer>"
    full = tokenizer.encode(prompt + row["answer"] + "<eos>")
    prompt_len = len(tokenizer.encode(prompt))
    mask = torch.ones(len(full), dtype=torch.float32)
    mask[:prompt_len] = 0
    return full[:-1], full[1:], mask[1:]


def collate(rows, tokenizer, max_len):
    samples = []
    for row in rows:
        x, y, mask = encode_record(row, tokenizer)
        samples.append((x[:max_len], y[:max_len], mask[:max_len]))
    length = max(item[0].size(0) for item in samples)
    xs, ys, masks = [], [], []
    for x, y, mask in samples:
        pad = length - x.size(0)
        xs.append(F.pad(x, (0, pad), value=tokenizer.pad_id))
        ys.append(F.pad(y, (0, pad), value=tokenizer.pad_id))
        masks.append(F.pad(mask, (0, pad)))
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


def evaluate(model, rows, tokenizer, device, batch, max_len, eos_weight):
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            x, y, mask = collate(rows[start:start + batch], tokenizer, max_len)
            logits = model(x.to(device))
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1), reduction="none").reshape_as(mask)
            weights = loss_weights(y.to(device), mask.to(device), tokenizer.eos_id, eos_weight)
            total += (losses * weights).sum().item()
            count += weights.sum().item()
    return total / max(1.0, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--valid-data-dir", required=True)
    parser.add_argument("--tokenizer-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--balanced-sources", action="store_true")
    parser.add_argument("--eos-weight", type=float, default=1.0)
    parser.add_argument("--scheduled-sampling", type=float, default=0.0)
    parser.add_argument("--init-checkpoint")
    args = parser.parse_args()
    prefix = Path(args.tokenizer_prefix)
    if not prefix.with_suffix(".model").exists():
        train_tokenizer(args.data_dir, prefix, args.vocab_size)
    tokenizer = SubwordTokenizer(prefix.with_suffix(".model"))
    train = load_records(args.data_dir)
    valid = load_records(args.valid_data_dir)
    if args.limit:
        train = train[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactSubwordCausalLM(tokenizer.vocab_size(), args.emb_dim, args.hidden_dim,
                                   args.layers, args.heads, args.max_len).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        source = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        target = model.state_dict()
        compatible = {name: value for name, value in source.items()
                      if name in target and target[name].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        print(f"initialized={args.init_checkpoint} tensors={len(compatible)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        sampler = None
        if args.balanced_sources:
            sampler = WeightedRandomSampler(source_balanced_weights(train), len(train), replacement=True)
        loader = DataLoader(train, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                            collate_fn=lambda rows: collate(rows, tokenizer, args.max_len))
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if args.scheduled_sampling > 0:
                predictions = logits.detach().argmax(dim=-1)
                mixed = scheduled_inputs(x, predictions, mask, args.scheduled_sampling)
                logits = model(mixed)
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").reshape_as(mask)
            weights = loss_weights(y, mask, tokenizer.eos_id, args.eos_weight)
            loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        valid_loss = evaluate(model, valid, tokenizer, device, args.batch, args.max_len, args.eos_weight)
        print(f"epoch={epoch} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "tokenizer": str(prefix.with_suffix(".model")),
                        "vocab_size": tokenizer.vocab_size(), "emb_dim": args.emb_dim,
                        "hidden_dim": args.hidden_dim, "layers": args.layers,
                        "heads": args.heads, "max_len": args.max_len,
                        "answer_id": tokenizer.answer_id, "eos_id": tokenizer.eos_id}, args.output)


if __name__ == "__main__":
    main()
