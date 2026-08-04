"""Train a compact act-conditioned question/answer dual encoder."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from conditioned_dual_selector import ConditionedDualEncoder
from hangul_semantic_plan import classify_response_act
from train_dual_selector import collate, load_records
from train_subword_lm import SubwordTokenizer


def act_map(rows):
    labels = sorted({classify_response_act(row["question"]) for row in rows})
    return {label: index for index, label in enumerate(labels)}


def metrics(model, queries, candidates, tokenizer, device, batch, max_len):
    model.eval()
    query_vec, candidate_vec = [], []
    with torch.no_grad():
        for start in range(0, len(queries), batch):
            q, a = collate(queries[start:start + batch], tokenizer, max_len)
            query_vec.append(model.encode(q.to(device)).cpu())
        for start in range(0, len(candidates), batch):
            q, a = collate(candidates[start:start + batch], tokenizer, max_len)
            candidate_vec.append(model.encode(a.to(device)).cpu())
    scores = torch.cat(query_vec) @ torch.cat(candidate_vec).T
    top = scores.argmax(dim=1).tolist()
    exact = sum(candidates[index]["answer"] == row["answer"]
                for index, row in zip(top, queries))
    prefix = sum(candidates[index]["answer"][:10] == row["answer"][:10]
                 for index, row in zip(top, queries))
    return {"rows": len(queries), "exact": exact, "prefix10": prefix}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer = SubwordTokenizer(args.tokenizer)
    train = load_records(args.train_dir)
    valid = load_records(args.valid_dir)
    if args.limit:
        train = train[:args.limit]
    labels = act_map(train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedDualEncoder(tokenizer.vocab_size(), len(labels),
                                   emb_dim=32, hidden_dim=64, output_dim=64,
                                   pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    best = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda rows: rows)
        for rows in loader:
            questions, answers = collate(rows, tokenizer, args.max_len)
            questions, answers = questions.to(device), answers.to(device)
            q_vec, a_vec, q_logits, a_logits = model.forward_with_act(questions, answers)
            labels_batch = torch.tensor(
                [labels[classify_response_act(row["question"])] for row in rows],
                device=device)
            contrastive = (F.cross_entropy(q_vec @ a_vec.T / 0.07, torch.arange(len(rows), device=device))
                           + F.cross_entropy((q_vec @ a_vec.T / 0.07).T, torch.arange(len(rows), device=device))) * 0.5
            loss = contrastive + 0.25 * (F.cross_entropy(q_logits, labels_batch) + F.cross_entropy(a_logits, labels_batch)) * 0.5
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        valid_metrics = metrics(model, valid, train, tokenizer, device, args.batch, args.max_len)
        print(f"epoch={epoch} loss={total / max(1, len(loader)):.4f} valid={json.dumps(valid_metrics)}")
        mean = total / max(1, len(loader))
        if mean < best:
            best = mean
            torch.save({"model": model.state_dict(), "tokenizer": args.tokenizer,
                        "vocab_size": tokenizer.vocab_size(), "num_acts": len(labels),
                        "act_labels": labels, "emb_dim": 32, "hidden_dim": 64,
                        "output_dim": 64, "pad_id": tokenizer.pad_id,
                        "max_len": args.max_len}, output)


if __name__ == "__main__":
    main()
