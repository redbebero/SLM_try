"""Evaluate a dual selector against a train-only answer bank."""

import argparse
import json
from pathlib import Path

import torch

from dual_selector import CompactDualEncoder
from syllable_core import SyllableTokenizer
from train_dual_selector import encode_all, load_records
from train_subword_lm import SubwordTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--tokenizer-type", choices=("subword", "syllable"), default="subword")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = (SubwordTokenizer(checkpoint["tokenizer"]) if args.tokenizer_type == "subword"
                 else SyllableTokenizer())
    vocab_size = (tokenizer.vocab_size() if args.tokenizer_type == "subword"
                  else tokenizer.get_vocab_size())
    model = CompactDualEncoder(
        vocab_size, checkpoint["emb_dim"], checkpoint["hidden_dim"],
        checkpoint["output_dim"], checkpoint["pad_id"],
    )
    model.load_state_dict(checkpoint["model"])
    candidate = load_records(args.candidate_dir)
    query = load_records(args.query_dir)
    if args.limit:
        query = query[:args.limit]
    device = torch.device("cpu")
    questions = encode_all(model, query, tokenizer, device, 256, checkpoint["max_len"])
    answers = encode_all(model, candidate, tokenizer, device, 256, checkpoint["max_len"], answer=True)
    top = (questions @ answers.T).argmax(dim=1).tolist()
    exact = sum(candidate[index]["answer"] == row["answer"] for index, row in zip(top, query))
    prefix = sum(candidate[index]["answer"][:10] == row["answer"][:10] for index, row in zip(top, query))
    report = {"rows": len(query), "exact": exact, "prefix10": prefix}
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
