"""Validate one-token JSONL records against their strict schema."""

from pathlib import Path
import argparse

from .validate_dataset import validate_dataset

DEFAULT_TOKEN_SCHEMA = Path("schemas/training-token-attribute.schema.json")


def validate_token_dataset(data_path: str | Path, schema_path: str | Path = DEFAULT_TOKEN_SCHEMA) -> list[str]:
    return validate_dataset(data_path, schema_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="?", default=Path("data/training-token-attributes.jsonl"))
    parser.add_argument("--schema", type=Path, default=DEFAULT_TOKEN_SCHEMA)
    args = parser.parse_args()
    errors = validate_token_dataset(args.data, args.schema)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"valid: {args.data}")


if __name__ == "__main__":
    main()
