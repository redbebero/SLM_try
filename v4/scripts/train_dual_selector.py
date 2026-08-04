"""Train a compact dual encoder on downloaded question-answer pairs."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dual_selector import CompactDualEncoder
from syllable_core import SyllableTokenizer
from train_subword_lm import SubwordTokenizer, load_records


def collate(rows, tokenizer, max_len):
    questions = [tokenizer.encode(row["question"])[:max_len] for row in rows]
    answers = [tokenizer.encode(row["answer"])[:max_len] for row in rows]
    q_len = max(1, max(item.numel() for item in questions))
    a_len = max(1, max(item.numel() for item in answers))
    q_batch = torch.full((len(rows), q_len), tokenizer.pad_id, dtype=torch.long)
    a_batch = torch.full((len(rows), a_len), tokenizer.pad_id, dtype=torch.long)
    for index, (question, answer) in enumerate(zip(questions, answers)):
        q_batch[index, :question.numel()] = question
        a_batch[index, :answer.numel()] = answer
    return q_batch, a_batch


def encode_all(model, rows, tokenizer, device, batch, max_len, answer=False):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            q, a = collate(rows[start:start + batch], tokenizer, max_len)
            values.append(model.encode((a if answer else q).to(device)).cpu())
    return torch.cat(values) if values else torch.empty((0, model.projection.out_features))


def retrieval_metrics(model, query_rows, candidate_rows, tokenizer, device, batch, max_len):
    questions = encode_all(model, query_rows, tokenizer, device, batch, max_len)
    answers = encode_all(model, candidate_rows, tokenizer, device, batch, max_len, answer=True)
    if not len(questions) or not len(answers):
        return {"exact": 0, "prefix10": 0, "rows": len(query_rows)}
    scores = questions @ answers.T
    top = scores.argmax(dim=1).tolist()
    exact = sum(candidate_rows[index]["answer"] == row["answer"] for index, row in zip(top, query_rows))
    prefix = sum(candidate_rows[index]["answer"][:10] == row["answer"][:10] for index, row in zip(top, query_rows))
    return {"exact": exact, "prefix10": prefix, "rows": len(query_rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer-type", choices=("subword", "syllable"), default="subword")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.tokenizer_type == "subword" and not args.tokenizer:
        parser.error("--tokenizer is required for --tokenizer-type subword")
    tokenizer = (SubwordTokenizer(args.tokenizer) if args.tokenizer_type == "subword"
                 else SyllableTokenizer())
    vocab_size = (tokenizer.vocab_size() if args.tokenizer_type == "subword"
                  else tokenizer.get_vocab_size())
    train = load_records(args.train_dir)
    valid = load_records(args.valid_dir)
    if args.limit:
        train = train[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactDualEncoder(
        vocab_size, args.emb_dim, args.hidden_dim, args.output_dim,
        tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(
            train, batch_size=args.batch, shuffle=True,
            collate_fn=lambda rows: collate(rows, tokenizer, args.max_len),
        )
        for questions, answers in loader:
            questions, answers = questions.to(device), answers.to(device)
            q_vec, a_vec = model(questions, answers)
            logits = q_vec @ a_vec.T / 0.07
            labels = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(1, len(loader))
        metrics = retrieval_metrics(model, valid, train, tokenizer, device, args.batch, args.max_len)
        print(f"epoch={epoch} train={train_loss:.4f} valid_against_train={json.dumps(metrics)}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "model": model.state_dict(), "tokenizer": args.tokenizer,
                "vocab_size": vocab_size, "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim, "output_dim": args.output_dim,
                "pad_id": tokenizer.pad_id, "max_len": args.max_len,
            }, output)


if __name__ == "__main__":
    main()
