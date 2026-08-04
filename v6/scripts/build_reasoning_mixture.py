import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build(paths, target):
    rows, ids = [], set()
    for source_name, path in paths:
        for row in read(path):
            if row["id"] in ids:
                continue
            ids.add(row["id"])
            row.setdefault("source", source_name)
            rows.append(row)
            if len(rows) == target:
                return rows
    if len(rows) < target:
        raise ValueError(f"only {len(rows)} unique rows available; need {target}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10000)
    for name in ("owned", "korquad", "gsm", "instruction", "existing"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = [(name, getattr(args, name)) for name in ("owned", "korquad", "gsm", "instruction", "existing")]
    rows = build(paths, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row.get("source", "unknown")] = counts.get(row.get("source", "unknown"), 0) + 1
    args.output.with_name("manifest.json").write_text(json.dumps({"rows": len(rows), "sources": counts}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows", counts)


if __name__ == "__main__":
    main()
