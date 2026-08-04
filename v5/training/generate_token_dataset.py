"""Generate one-token JSONL dataset from the reviewed lexicon."""

import argparse
from pathlib import Path

from .token_data import lexicon_to_records, load_lexicon, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/token-lexicon.json"))
    parser.add_argument("--output", type=Path, default=Path("data/training-token-attributes.jsonl"))
    args = parser.parse_args()
    records = lexicon_to_records(load_lexicon(args.input))
    write_jsonl(records, args.output)
    print({"records": len(records), "output": str(args.output)})


if __name__ == "__main__":
    main()
