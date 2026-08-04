"""Small baseline fine-tune for the Korean KoGPT2 teacher."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def load_rows(path, limit=0):
    rows = [json.loads(line) for line in (Path(path) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def encode(row, tokenizer, max_len):
    prompt = f"Q: {row['question']}\nA: "
    text = prompt + row["answer"] + tokenizer.eos_token
    full = tokenizer(text, add_special_tokens=False)["input_ids"][:max_len]
    prompt_len = min(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]), len(full))
    labels = [-100] * prompt_len + full[prompt_len:]
    return torch.tensor(full, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate(rows, tokenizer, max_len):
    items = [encode(row, tokenizer, max_len) for row in rows]
    length = max(item[0].numel() for item in items)
    ids, labels = [], []
    for input_ids, target in items:
        pad = length - input_ids.numel()
        ids.append(torch.nn.functional.pad(input_ids, (0, pad), value=tokenizer.pad_token_id))
        labels.append(torch.nn.functional.pad(target, (0, pad), value=-100))
    return {"input_ids": torch.stack(ids), "labels": torch.stack(labels),
            "attention_mask": torch.stack([item.ne(tokenizer.pad_token_id) for item in ids]).long()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.model) / "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = load_rows(args.data_dir, args.limit)
    loader = DataLoader(rows, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer, args.max_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch} train={total / max(1, len(loader)):.4f}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
