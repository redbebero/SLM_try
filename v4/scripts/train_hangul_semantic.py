"""Train the compact Hangul semantic encoder on downloaded data only."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hangul_semantic_encoder import HangulSemanticEncoder, count_parameters
from tokenizer import KoJamoTokenizer


def load_records(directory):
    path = Path(directory) / "records.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pad_batch(texts, tokenizer, max_len, device):
    encoded = [tokenizer.encode(text)[:max_len] for text in texts]
    batch = torch.zeros(len(encoded), max_len, 6, dtype=torch.long, device=device)
    mask = torch.zeros(len(encoded), max_len, dtype=torch.bool, device=device)
    for index, item in enumerate(encoded):
        length = min(item.size(0), max_len)
        if length:
            batch[index, :length] = item[:length].to(device)
            mask[index, :length] = True
    return batch, mask


def category_map(records):
    labels = sorted({row.get("category", "unknown") for row in records})
    return {label: index for index, label in enumerate(labels)}


def contrastive_loss(question, answer, temperature=0.07):
    logits = question @ answer.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) * 0.5


def same_category_batches(rows, batch_size, seed):
    """Make in-batch negatives share a category, making them harder."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get("category", "unknown")].append(index)
    rng = random.Random(seed)
    batches = []
    for indices in groups.values():
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            if len(batch) > 1:
                batches.append(batch)
    rng.shuffle(batches)
    return batches


def collate(rows, tokenizer, max_len, label_map, device):
    questions, q_mask = pad_batch([row["question"] for row in rows], tokenizer, max_len, device)
    answers, a_mask = pad_batch([row["answer"] for row in rows], tokenizer, max_len, device)
    labels = torch.tensor([label_map.get(row.get("category", "unknown"), 0)
                           for row in rows], dtype=torch.long, device=device)
    return questions, q_mask, answers, a_mask, labels


def evaluate(model, rows, candidates, tokenizer, label_map, device, batch, max_len):
    model.eval()
    query_vectors, candidate_vectors = [], []
    query_labels, candidate_labels = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            part = rows[start:start + batch]
            q, qm = pad_batch([r["question"] for r in part], tokenizer, max_len, device)
            qv, ql = model(q, qm)
            query_vectors.append(qv.cpu())
            query_labels.extend(label_map.get(r.get("category", "unknown"), 0) for r in part)
        for start in range(0, len(candidates), batch):
            part = candidates[start:start + batch]
            a, am = pad_batch([r["answer"] for r in part], tokenizer, max_len, device)
            av, al = model(a, am)
            candidate_vectors.append(av.cpu())
            candidate_labels.extend(label_map.get(r.get("category", "unknown"), 0) for r in part)
    questions = torch.cat(query_vectors)
    answers = torch.cat(candidate_vectors)
    scores = questions @ answers.T
    top10 = scores.topk(min(10, scores.size(1)), dim=1).indices
    top1 = top10[:, 0]
    # A response is considered the paired positive only when its raw answer
    # matches; duplicate answers are allowed to count as a valid hit.
    answer_texts = [row["answer"] for row in candidates]
    exact1 = sum(answer_texts[index] == row["answer"]
                for index, row in zip(top1.tolist(), rows))
    exact10 = sum(any(answer_texts[index] == row["answer"] for index in indices.tolist())
                  for row, indices in zip(rows, top10))
    category_top1 = sum(candidate_labels[index] == label
                        for index, label in zip(top1.tolist(), query_labels))
    return {
        "rows": len(rows), "exact1": exact1 / max(1, len(rows)),
        "exact10": exact10 / max(1, len(rows)),
        "category_top1": category_top1 / max(1, len(rows)),
    }


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KoJamoTokenizer()
    train_rows = load_records(args.train_dir)
    valid_rows = load_records(args.valid_dir)
    test_rows = load_records(args.test_dir) if args.test_dir else []
    if args.limit:
        train_rows = train_rows[:args.limit]
    label_map = category_map(train_rows)
    model = HangulSemanticEncoder(
        num_categories=len(label_map), emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
    ).to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location=device)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        if args.hard_negative:
            loader = DataLoader(
                train_rows,
                batch_sampler=same_category_batches(
                    train_rows, args.batch, args.seed + epoch),
                collate_fn=lambda rows: collate(
                    rows, tokenizer, args.max_len, label_map, device),
            )
        else:
            loader = DataLoader(
                train_rows, batch_size=args.batch, shuffle=True,
                collate_fn=lambda rows: collate(
                    rows, tokenizer, args.max_len, label_map, device),
            )
        for questions, q_mask, answers, a_mask, labels in loader:
            q_vec, q_category = model(questions, q_mask)
            a_vec, a_category = model(answers, a_mask)
            loss = contrastive_loss(q_vec, a_vec)
            loss = loss + args.category_weight * (
                F.cross_entropy(q_category, labels) +
                F.cross_entropy(a_category, labels)) * 0.5
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        metrics = evaluate(model, valid_rows, train_rows, tokenizer, label_map,
                           device, args.batch, args.max_len)
        mean_loss = total / max(1, len(loader))
        print(f"epoch={epoch} loss={mean_loss:.4f} valid={json.dumps(metrics)}")
        if mean_loss < best:
            best = mean_loss
            torch.save({
                "model": model.state_dict(), "label_map": label_map,
                "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim,
                "output_dim": args.output_dim, "max_len": args.max_len,
                "parameters": count_parameters(model),
            }, output)
    if test_rows:
        checkpoint = torch.load(output, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print("test=" + json.dumps(evaluate(
            model, test_rows, train_rows, tokenizer, label_map,
            device, args.batch, args.max_len)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--init", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--category-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hard-negative", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
