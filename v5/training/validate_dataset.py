"""Validate JSONL training records against the project schema."""

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


DEFAULT_SCHEMA = Path("schemas/training-spell-record.schema.json")


def validate_dataset(
    data_path: str | Path,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    errors: list[str] = []
    for line_number, line in enumerate(Path(data_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON: {error.msg}")
            continue
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
            path = ".".join(str(part) for part in error.path) or "$"
            errors.append(f"line {line_number}: {path}: {error.validator}: {error.message}")
    if not errors and not Path(data_path).read_text(encoding="utf-8").strip():
        errors.append("dataset: must contain at least one record")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="?", default=Path("data/training-spells.jsonl"))
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    errors = validate_dataset(args.data, args.schema)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"valid: {args.data}")


if __name__ == "__main__":
    main()
