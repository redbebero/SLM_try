"""Train a compact denoising seq2seq model on downloaded Korean records."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from denoising_seq2seq import corrupt_tokens, count_parameters
from seq2seq_lm import CompactSeq2SeqLM
from train_subword_lm import SubwordTokenizer


def load_texts(directory):
    rows = [json.loads(line) for line in (Path(directory) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = []
    for row in rows:
        texts.extend((row["question"], row["answer"]))
    return list(dict.fromkeys(texts))


def encode(text, tokenizer, max_len):
    original = tokenizer.encode(text)[:max_len - 1]
    target = torch.cat([original, torch.tensor([tokenizer.eos_id])])
    source = corrupt_tokens(original, mask_id=tokenizer.processor.unk_id())
    decoder_input = torch.cat([torch.tensor([tokenizer.answer_id]), target[:-1]])
    return source, decoder_input, target


def collate(texts, tokenizer, max_len):
    items = [encode(text, tokenizer, max_len) for text in texts]
    length = max(len(item[0]) for item in items)
    sources, decoders, targets = [], [], []
    for source, decoder, target in items:
        sources.append(F.pad(source, (0, length - len(source)), value=tokenizer.pad_id))
        decoders.append(F.pad(decoder, (0, length - len(decoder)), value=tokenizer.pad_id))
        targets.append(F.pad(target, (0, length - len(target)), value=tokenizer.pad_id))
    return torch.stack(sources), torch.stack(decoders), torch.stack(targets)


def loss_on(model, texts, tokenizer, device, batch, max_len):
    model.eval()
    total = count = 0.0
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            source, decoder, target = collate(texts[start:start + batch], tokenizer, max_len)
            logits = model(source.to(device), decoder.to(device))
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.to(device).reshape(-1), reduction="none")
            mask = target.ne(tokenizer.pad_id).to(losses.dtype).reshape(-1)
            total += (losses * mask).sum().item()
            count += mask.sum().item()
    return total / max(1.0, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer = SubwordTokenizer(args.tokenizer)
    train = load_texts(args.train_dir)
    valid = load_texts(args.valid_dir)
    if args.limit:
        train = train[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactSeq2SeqLM(tokenizer.vocab_size(), emb_dim=16, hidden_dim=64,
                             layers=1, heads=4, max_src_len=args.max_len,
                             max_tgt_len=args.max_len, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda rows: collate(rows, tokenizer, args.max_len))
        for source, decoder, target in loader:
            source, decoder, target = source.to(device), decoder.to(device), target.to(device)
            logits = model(source, decoder)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        valid_loss = loss_on(model, valid, tokenizer, device, args.batch, args.max_len)
        print(f"epoch={epoch} train={total / max(1, len(loader)):.4f} valid={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "tokenizer": args.tokenizer,
                        "vocab_size": tokenizer.vocab_size(), "emb_dim": 16,
                        "hidden_dim": 64, "layers": 1, "heads": 4,
                        "max_src_len": args.max_len, "max_tgt_len": args.max_len,
                        "pad_id": tokenizer.pad_id, "answer_id": tokenizer.answer_id,
                        "eos_id": tokenizer.eos_id, "parameters": count_parameters(model)}, output)
    print(f"train_rows={len(train)} valid_rows={len(valid)} parameters={count_parameters(model)}")


if __name__ == "__main__":
    main()
