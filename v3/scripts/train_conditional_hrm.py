"""Train the prompt-conditioned six-track HRM decoder experiment."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import KoJamoDataset
from hrm_model import HRMConditionalNet
from train_hrm import masked_loss, sft_collate_fn, build_scheduled_input


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KoJamoDataset(data_dir=args.data_dir, seq_length=1000, stride=100, is_sft=True)
    train_size = int(len(dataset) * 0.9)
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    collate = lambda batch: sft_collate_fn(batch, max_seq_length=args.max_seq_length)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch, shuffle=False,
                            collate_fn=collate)
    model = HRMConditionalNet(
        vocab_sizes=dataset.tokenizer.get_vocab_sizes(), emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim, cycle_steps=args.cycle_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for x, y, mask in tqdm(train_loader, desc=f"Conditional HRM Epoch {epoch}"):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            # x[t] is the previous token for y[t]. It is prompt-side through
            # the first answer target and answer-side afterwards.
            answer_input = torch.zeros_like(mask, dtype=torch.bool)
            answer_input[:, 1:] = mask[:, :-1] > 0
            prompt_mask = (~answer_input) & (x.sum(dim=-1) > 0)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.forward_segment(x, prompt_mask=prompt_mask)
            teacher_loss = masked_loss(logits, y, mask)
            rollout_x = build_scheduled_input(x, logits, mask, args.scheduled_sampling)
            rollout_logits, _ = model.forward_segment(rollout_x, prompt_mask=prompt_mask)
            rollout_loss = masked_loss(rollout_logits, y, mask)
            loss = (teacher_loss + rollout_loss) * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item()
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                answer_input = torch.zeros_like(mask, dtype=torch.bool)
                answer_input[:, 1:] = mask[:, :-1] > 0
                prompt_mask = (~answer_input) & (x.sum(dim=-1) > 0)
                logits, _ = model.forward_segment(x, prompt_mask=prompt_mask)
                val_total += masked_loss(logits, y, mask).item()
        train_avg = train_total / max(1, len(train_loader))
        val_avg = val_total / max(1, len(val_loader))
        print(f"Conditional HRM Epoch {epoch} | train={train_avg:.4f} | val={val_avg:.4f}")
        if val_avg < best:
            best = val_avg
            torch.save({
                "model": model.state_dict(),
                "conditional_decoder": True,
                "sft_format": True,
            }, args.output)
            print(f"Conditional HRM best saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="train_data_hrm_dialogue_pure_v2")
    parser.add_argument("--output", default="checkpoints/hrm_conditional_pure_v2_best.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cycle-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scheduled-sampling", type=float, default=0.5)
    run(parser.parse_args())
