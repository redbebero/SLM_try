"""Evaluate compact seq2seq free-running generation."""

import argparse
import json
from pathlib import Path

import torch

from seq2seq_lm import CompactSeq2SeqLM
from train_seq2seq_lm import load_records
from train_subword_lm import SubwordTokenizer


def generate(model, tokenizer, question, max_new_tokens):
    device = next(model.parameters()).device
    source_ids = tokenizer.encode(f"Q: {question}")[:model.max_src_len]
    source = source_ids.unsqueeze(0).to(device)
    decoder = torch.tensor([[tokenizer.answer_id]], device=device)
    with torch.no_grad():
        for _ in range(min(max_new_tokens, model.max_tgt_len - 1)):
            logits = model(source, decoder)[:, -1]
            next_id = int(torch.argmax(logits, dim=-1).item())
            decoder = torch.cat([decoder, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tokenizer.eos_id:
                break
    ids = decoder[0, 1:].tolist()
    if tokenizer.eos_id in ids:
        ids = ids[:ids.index(tokenizer.eos_id)]
    return tokenizer.decode(ids), tokenizer.eos_id in decoder[0].tolist()


def beam_generate(model, tokenizer, question, max_new_tokens, beam_size):
    device = next(model.parameters()).device
    source_ids = tokenizer.encode(f"Q: {question}")[:model.max_src_len]
    source = source_ids.unsqueeze(0).to(device)
    beams = [(torch.tensor([tokenizer.answer_id], device=device), 0.0, False)]
    with torch.no_grad():
        for _ in range(min(max_new_tokens, model.max_tgt_len - 1)):
            candidates = []
            active = [beam for beam in beams if not beam[2]]
            finished = [beam for beam in beams if beam[2]]
            for decoder, score, ended in active:
                logits = model(source, decoder.unsqueeze(0))[:, -1]
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                values, indices = torch.topk(log_probs, beam_size)
                for value, index in zip(values.tolist(), indices.tolist()):
                    token = torch.tensor([index], device=device)
                    sequence = torch.cat([decoder, token])
                    candidates.append((sequence, score + value, index == tokenizer.eos_id))
            beams = sorted(finished + candidates, key=lambda item: item[1] / max(1, item[0].numel() - 1), reverse=True)[:beam_size]
            if all(item[2] for item in beams):
                break
    best = max(beams, key=lambda item: item[1] / max(1, item[0].numel() - 1))
    ids = best[0][1:].tolist()
    if tokenizer.eos_id in ids:
        ids = ids[:ids.index(tokenizer.eos_id)]
    return tokenizer.decode(ids), best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--beam-size", type=int, default=1)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = SubwordTokenizer(checkpoint["tokenizer"])
    model = CompactSeq2SeqLM(checkpoint["vocab_size"], checkpoint["emb_dim"], checkpoint["hidden_dim"],
                             checkpoint["layers"], checkpoint["heads"], checkpoint["max_src_len"],
                             checkpoint["max_tgt_len"], checkpoint["pad_id"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = load_records(args.data_dir)[:args.limit]
    exact = prefix = eos = 0
    for row in rows:
        if args.beam_size > 1:
            output, ended = beam_generate(model, tokenizer, row["question"], args.max_new_tokens, args.beam_size)
        else:
            output, ended = generate(model, tokenizer, row["question"], args.max_new_tokens)
        exact += output == row["answer"]
        prefix += output[:10] == row["answer"][:10]
        eos += ended
        print(json.dumps({"question": row["question"], "answer": row["answer"], "output": output, "eos": ended}, ensure_ascii=False))
    total = max(1, len(rows))
    print(f"summary exact={exact}/{total} prefix10={prefix}/{total} eos={eos}/{total}")


if __name__ == "__main__":
    main()
