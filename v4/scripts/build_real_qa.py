"""Extract question/explanation pairs from the downloaded Korean corpus."""

from pathlib import Path


def extract(path):
    rows = []
    seen = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if "?" not in raw:
            continue
        question, answer = raw.split("?", 1)
        question, answer = question.strip() + "?", answer.strip()
        if not 10 <= len(question) <= 60 or not 10 <= len(answer) <= 80:
            continue
        row = (question, answer)
        if row not in seen:
            seen.add(row); rows.append(row)
    return rows


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(f"{q}\t{a}" for q, a in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    all_rows = extract("train_data_clean/pretrain_train.txt")
    split = max(1, int(len(all_rows) * 0.8))
    train, valid = all_rows[:split], all_rows[split:]
    write("experiments/real_qa/train/train.tsv", train)
    write("experiments/real_qa/valid/valid.tsv", valid)
    print(f"all={len(all_rows)} train={len(train)} valid={len(valid)}")
