"""Train compact GRU question-to-answer generation on downloaded SFT rows."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gru_seq2seq import CompactGRUSeq2Seq
from train_subword_lm import SubwordTokenizer, load_records


def scheduled_decoder_inputs(decoder, predictions, probability):
    if probability <= 0:
        return decoder
    mixed = decoder.clone()
    replace = torch.rand_like(decoder[:, 1:], dtype=torch.float32) < probability
    mixed[:, 1:] = torch.where(replace, predictions[:, :-1].detach(), mixed[:, 1:])
    return mixed


def encode_row(row, tokenizer, max_src, max_tgt):
    source = tokenizer.encode(row["question"])[:max_src]
    target = tokenizer.encode(row["answer"] + "<eos>")[:max_tgt]
    decoder = torch.cat([torch.tensor([tokenizer.answer_id]), target[:-1]])
    return source, decoder, target


def collate(rows, tokenizer, max_src, max_tgt):
    items = [encode_row(row, tokenizer, max_src, max_tgt) for row in rows]
    src_len = max(1, max(item[0].numel() for item in items))
    tgt_len = max(1, max(item[2].numel() for item in items))
    sources = torch.full((len(rows), src_len), tokenizer.pad_id, dtype=torch.long)
    decoders = torch.full((len(rows), tgt_len), tokenizer.pad_id, dtype=torch.long)
    targets = torch.full((len(rows), tgt_len), tokenizer.pad_id, dtype=torch.long)
    for index, (source, decoder, target) in enumerate(items):
        sources[index, :source.numel()] = source
        decoders[index, :decoder.numel()] = decoder
        targets[index, :target.numel()] = target
    return sources, decoders, targets


def loss_for(model, rows, tokenizer, device, batch, max_src, max_tgt):
    model.eval()
    total = count = 0.0
    with torch.no_grad():
        loader = DataLoader(rows, batch_size=batch, shuffle=False,
                            collate_fn=lambda values: collate(values, tokenizer, max_src, max_tgt))
        for source, decoder, target in loader:
            logits, _ = model(source.to(device), decoder.to(device))
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.to(device).reshape(-1),
                ignore_index=tokenizer.pad_id, reduction="sum",
            )
            total += losses.item()
            count += target.ne(tokenizer.pad_id).sum().item()
    return total / max(1, count)


def generate(model, tokenizer, question, device, max_new=80):
    model.eval()
    source = tokenizer.encode(question).unsqueeze(0).to(device)
    decoder = torch.tensor([[tokenizer.answer_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        memory, hidden = model.encode(source)
        keys = model.key(memory)
        source_mask = source.ne(model.pad_id)
        for _ in range(max_new):
            logits, hidden, _ = model.decode_step(
                decoder[:, -1], hidden, memory, keys, source_mask
            )
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            decoder = torch.cat([decoder, next_id], dim=1)
            if int(next_id.item()) == tokenizer.eos_id:
                break
    return tokenizer.decode(decoder[0, 1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--valid-data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-src", type=int, default=96)
    parser.add_argument("--max-tgt", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scheduled-sampling", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = SubwordTokenizer(args.tokenizer)
    train, valid = load_records(args.data_dir), load_records(args.valid_data_dir)
    if args.limit:
        train = train[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactGRUSeq2Seq(tokenizer.vocab_size(), args.emb_dim, args.hidden_dim,
                              tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda values: collate(values, tokenizer, args.max_src, args.max_tgt))
        for source, decoder, target in loader:
            source, decoder, target = source.to(device), decoder.to(device), target.to(device)
            logits, _ = model(source, decoder)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                   ignore_index=tokenizer.pad_id)
            if args.scheduled_sampling > 0:
                mixed = scheduled_decoder_inputs(decoder, logits.argmax(dim=-1), args.scheduled_sampling)
                rollout_logits, _ = model(source, mixed)
                rollout_loss = F.cross_entropy(
                    rollout_logits.reshape(-1, rollout_logits.size(-1)), target.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
                loss = (loss + rollout_loss) * 0.5
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        valid_loss = loss_for(model, valid, tokenizer, device, args.batch, args.max_src, args.max_tgt)
        print(f"epoch={epoch} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            torch.save({
                "model": model.state_dict(), "tokenizer": args.tokenizer,
                "vocab_size": tokenizer.vocab_size(), "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim, "pad_id": tokenizer.pad_id,
                "answer_id": tokenizer.answer_id, "eos_id": tokenizer.eos_id,
                "max_src": args.max_src, "max_tgt": args.max_tgt,
            }, output)
    for row in valid[:3]:
        print(json.dumps({"question": row["question"], "answer": row["answer"],
                          "output": generate(model, tokenizer, row["question"], device)},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
