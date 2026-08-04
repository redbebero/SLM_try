"""Train the compact question-encoder/answer-decoder model."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seq2seq_lm import CompactSeq2SeqLM
from train_subword_lm import SubwordTokenizer


def load_records(path):
    return [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_record(row, tokenizer, max_src_len, max_tgt_len):
    source = tokenizer.encode(f"Q: {row['question']}")[:max_src_len]
    target = tokenizer.encode(row["answer"] + "<eos>")[:max_tgt_len]
    decoder_input = torch.cat([torch.tensor([tokenizer.answer_id]), target[:-1]])
    return source, decoder_input, target


def collate(rows, tokenizer, max_src_len, max_tgt_len):
    items = [encode_record(row, tokenizer, max_src_len, max_tgt_len) for row in rows]
    src_len = max(item[0].size(0) for item in items)
    tgt_len = max(item[2].size(0) for item in items)
    sources, decoder_inputs, targets = [], [], []
    for source, decoder_input, target in items:
        sources.append(F.pad(source, (0, src_len - source.size(0)), value=tokenizer.pad_id))
        decoder_inputs.append(F.pad(decoder_input, (0, tgt_len - decoder_input.size(0)), value=tokenizer.pad_id))
        targets.append(F.pad(target, (0, tgt_len - target.size(0)), value=tokenizer.pad_id))
    return torch.stack(sources), torch.stack(decoder_inputs), torch.stack(targets)


def evaluate(model, rows, tokenizer, device, batch, max_src_len, max_tgt_len):
    model.eval()
    total, count = 0.0, 0
    loader = DataLoader(rows, batch_size=batch, shuffle=False,
                        collate_fn=lambda x: collate(x, tokenizer, max_src_len, max_tgt_len))
    with torch.no_grad():
        for source, decoder_input, target in loader:
            logits = model(source.to(device), decoder_input.to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.to(device).reshape(-1),
                                   ignore_index=tokenizer.pad_id, reduction="sum")
            total += loss.item()
            count += int(target.ne(tokenizer.pad_id).sum())
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--valid-data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-src-len", type=int, default=128)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer = SubwordTokenizer(args.tokenizer)
    train, valid = load_records(args.data_dir), load_records(args.valid_data_dir)
    if args.limit:
        train = train[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactSeq2SeqLM(tokenizer.vocab_size(), args.emb_dim, args.hidden_dim,
                             args.layers, args.heads, args.max_src_len, args.max_tgt_len,
                             tokenizer.pad_id).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        source = checkpoint.get("model", checkpoint)
        compatible = {name: value for name, value in source.items()
                      if name in model.state_dict() and model.state_dict()[name].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        print(f"initialized={args.init_checkpoint} tensors={len(compatible)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda x: collate(x, tokenizer, args.max_src_len, args.max_tgt_len))
        for source, decoder_input, target in loader:
            source, decoder_input, target = source.to(device), decoder_input.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, decoder_input)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                   ignore_index=tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        valid_loss = evaluate(model, valid, tokenizer, device, args.batch, args.max_src_len, args.max_tgt_len)
        print(f"epoch={epoch} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "tokenizer": str(args.tokenizer),
                        "vocab_size": tokenizer.vocab_size(), "emb_dim": args.emb_dim,
                        "hidden_dim": args.hidden_dim, "layers": args.layers,
                        "heads": args.heads, "max_src_len": args.max_src_len,
                        "max_tgt_len": args.max_tgt_len, "pad_id": tokenizer.pad_id,
                        "answer_id": tokenizer.answer_id, "eos_id": tokenizer.eos_id}, args.output)


if __name__ == "__main__":
    main()
