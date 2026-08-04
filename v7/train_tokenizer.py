import argparse, json
from pathlib import Path
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/tokenizer"))
    ap.add_argument("--vocab-size", type=int, default=4_096)
    args = ap.parse_args()
    args.input = args.input or (Path("data/processed/train_expanded.txt") if Path("data/processed/train_expanded.txt").exists() else Path("data/processed/train.txt"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extras = [p for p in (Path("data/chat_seed.jsonl"), Path("data/chat_train.jsonl"), Path("data/chat_real.jsonl"), Path("data/tokenizer_extra.txt")) if p.exists()]
    corpus = ",".join([str(args.input), *(str(p) for p in extras)])
    required = "".join(sorted(set("".join(p.read_text(encoding="utf-8") for p in extras)) - set(" \t\r\n")))
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=str(args.out), model_type="unigram", vocab_size=args.vocab_size, character_coverage=1.0, input_sentence_size=5000, max_sentence_length=4192, hard_vocab_limit=False, required_chars=required, pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols="<|system|>,<|user|>,<|assistant|>,<|end|>")
    p = spm.SentencePieceProcessor(model_file=str(args.out) + ".model")
    sample = args.input.read_text(encoding="utf-8").splitlines()[0]
    ids = p.encode(sample, out_type=int)
    Path(str(args.out) + ".check.json").write_text(json.dumps({"text": sample, "ids": ids, "decoded": p.decode(ids), "round_trip": p.decode(ids) == sample}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"vocab_size": p.vocab_size(), "round_trip": p.decode(ids) == sample}, ensure_ascii=False))


if __name__ == "__main__":
    main()
