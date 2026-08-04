"""Build disjoint public corpora for a from-scratch Korean-jamo HRM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from random import Random


BAD = re.compile(
    r"(https?://|www\.|인공지능\s*(?:언어\s*)?(?:모델|시스템|프로그램)"
    r"|실제로\s*(?:여행|경험|할\s*수\s*없)|�|질문\s*:|답변\s*:)",
    re.I,
)


def clean(value: str, minimum: int = 2, maximum: int = 420) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if not minimum <= len(value) <= maximum or BAD.search(value):
        return ""
    if len(re.findall(r"[가-힣]", value)) < 2:
        return ""
    if re.search(r"(.{4,})\s*\1", value):
        return ""
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = []
        for message in row.get("messages", []):
            content = clean(message.get("content", ""))
            if content and message.get("role") in {"user", "assistant"}:
                messages.append({"role": message["role"], "content": content})
        answer = clean(row.get("answer", ""), minimum=8)
        if messages and messages[-1]["role"] == "user" and answer:
            rows.append({"messages": messages, "answer": answer})
    return rows


def to_sft(row: dict) -> str:
    lines = []
    for message in row["messages"]:
        speaker = "사용자" if message["role"] == "user" else "상대"
        lines.append(f"{speaker}: {message['content']}")
    return f"Q: {' '.join(lines)}\nA: {row['answer']}"


def prompt_key(row: dict) -> str:
    """Stable key for the user-side context, independent of the answer."""
    prompt = "\n".join(
        message["content"] for message in row["messages"]
        if message["role"] == "user"
    )
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def unique(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        # Keep one answer per user context. Multiple answers to one prompt
        # make validation look better while teaching memorized responses.
        key = prompt_key(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def write_dir(path: Path, rows: list[dict], sft: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sft:
        text = "\n\n".join(to_sft(row) for row in rows) + "\n"
    else:
        lines = []
        for row in rows:
            lines.extend(message["content"] for message in row["messages"])
            lines.append(row["answer"])
        # Pack short public turns into longer independent documents so the
        # pretraining loader can form real 256-token next-token windows.
        packed = []
        for index in range(0, len(lines), 24):
            packed.append(" ".join(lines[index:index + 24]))
        lines = packed
        text = "\n".join(lines) + "\n"
    (path / ("sft.txt" if sft else "text.txt")).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empathetic-train", type=Path, required=True)
    parser.add_argument("--empathetic-valid", type=Path, required=True)
    parser.add_argument("--roleplay-train", type=Path, required=True)
    parser.add_argument("--roleplay-valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roleplay-limit", type=int, default=6000)
    parser.add_argument("--valid-roleplay-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emp_train = load_jsonl(args.empathetic_train)
    emp_valid = load_jsonl(args.empathetic_valid)
    role_train = load_jsonl(args.roleplay_train)
    role_valid = load_jsonl(args.roleplay_valid)
    rng = Random(args.seed)
    rng.shuffle(emp_train); rng.shuffle(emp_valid)
    rng.shuffle(role_train); rng.shuffle(role_valid)
    train = unique(emp_train + role_train[:args.roleplay_limit])
    valid = unique(emp_valid + role_valid[:args.valid_roleplay_limit])
    valid_keys = {prompt_key(row) for row in valid}
    train = [row for row in train if prompt_key(row) not in valid_keys]
    write_dir(args.output / "sft_train", train, True)
    write_dir(args.output / "sft_valid", valid, True)
    write_dir(args.output / "pretrain_train", train, False)
    write_dir(args.output / "pretrain_valid", valid, False)
    manifest = {
        "sources": [str(args.empathetic_train), str(args.empathetic_valid),
                    str(args.roleplay_train), str(args.roleplay_valid)],
        "train": len(train), "valid": len(valid), "seed": args.seed,
        "question_overlap": len({prompt_key(row) for row in train}
                                 & {prompt_key(row) for row in valid}),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
