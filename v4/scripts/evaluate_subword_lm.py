"""Evaluate compact SentencePiece causal SFT generation."""

import argparse
import json
from pathlib import Path

import torch

from subword_lm import CompactSubwordCausalLM
from train_subword_lm import SubwordTokenizer, load_records
from retrieval import TfidfCharNgramRetriever


def generate(model, tokenizer, question, max_new_tokens=80, repetition_penalty=1.0, temperature=0.0, top_k=0):
    prompt = tokenizer.encode(f"Q: {question}<answer>").unsqueeze(0)
    device = next(model.parameters()).device
    ids = prompt.to(device)
    seen = set()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.max_len:])[:, -1, :]
            if repetition_penalty > 1.0:
                for token_id in seen:
                    value = logits[0, token_id]
                    logits[0, token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
            if temperature > 0:
                scaled = logits / temperature
                if top_k > 0:
                    values, indices = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1)
                    probs = torch.softmax(values, dim=-1)
                    next_id = int(indices[0, torch.multinomial(probs[0], 1)].item())
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    next_id = int(torch.multinomial(probs[0], 1).item())
            else:
                next_id = int(torch.argmax(logits, dim=-1).item())
            seen.add(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tokenizer.eos_id:
                break
    generated = ids[0, prompt.size(1):].tolist()
    eos = generated.index(tokenizer.eos_id) if tokenizer.eos_id in generated else len(generated)
    return tokenizer.decode(generated[:eos]), tokenizer.eos_id in generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--retrieval-train-dir")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    torch.manual_seed(args.seed)
    tokenizer = SubwordTokenizer(checkpoint["tokenizer"])
    model = CompactSubwordCausalLM(
        checkpoint["vocab_size"], checkpoint["emb_dim"], checkpoint["hidden_dim"],
        checkpoint["layers"], checkpoint["heads"], checkpoint["max_len"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = load_records(args.data_dir)[:args.limit]
    retriever = TfidfCharNgramRetriever(load_records(args.retrieval_train_dir)) if args.retrieval_train_dir else None
    exact = prefix = eos_count = 0
    for row in rows:
        question = row["question"]
        if retriever:
            hit = retriever.search(question, top_k=1, source=row.get("source"), category=row.get("category"))[0]
            question = f"참고 질문: {hit['question']}\n참고 답변: {hit['answer']}\n현재 질문: {question}"
        output, has_eos = generate(model, tokenizer, question, args.max_new_tokens, args.repetition_penalty, args.temperature, args.top_k)
        exact += output == row["answer"]
        prefix += output[:10] == row["answer"][:10]
        eos_count += has_eos
        print(json.dumps({"question": row["question"], "answer": row["answer"], "output": output, "eos": has_eos}, ensure_ascii=False))
    total = max(1, len(rows))
    print(f"summary exact={exact}/{total} prefix10={prefix}/{total} eos={eos_count}/{total}")


if __name__ == "__main__":
    main()
