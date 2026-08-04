"""Extract short final answers from the downloaded Korean math QA corpus."""

import re
from pathlib import Path

UNITS = r"살|세|달러|페이지|일|개|원|분|시간|명|킬로미터|리터|센트|초|개월|년|도"
RESULT_UNITS = {"살", "세", "달러", "페이지", "일", "개", "원", "분", "시간", "명", "킬로미터", "리터", "센트", "초", "도"}


def normalize(explanation):
    text = explanation.strip()
    matches = re.findall(rf"\$?(\d[\d,]*(?:\.\d+)?)\s*({UNITS})?", text)
    if not matches:
        return ""
    preferred = [match for match in matches if match[1] in RESULT_UNITS]
    number, unit = (preferred or matches)[-1]
    number = number.replace(",", "")
    if number.startswith("0") and len(number) > 1 and not number.startswith("0."):
        number = number.lstrip("0") or "0"
    return f"{number}{unit or ''}입니다."


def main():
    rows = []
    seen = set()
    for raw in Path("train_data_clean/pretrain_train.txt").read_text(encoding="utf-8").splitlines():
        if "?" not in raw:
            continue
        question, explanation = raw.split("?", 1)
        question = question.strip() + "?"
        answer = normalize(explanation)
        if not 10 <= len(question) <= 120 or not answer:
            continue
        row = (question, answer)
        if row not in seen:
            seen.add(row); rows.append(row)
    split = max(1, int(len(rows) * 0.8))
    root = Path("experiments/normalized_qa")
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "valid").mkdir(parents=True, exist_ok=True)
    for name, data in [("train", rows[:split]), ("valid", rows[split:])]:
        (root / name / f"{name}.tsv").write_text("\n".join(f"{q}\t{a}" for q, a in data) + "\n", encoding="utf-8")
    print(f"all={len(rows)} train={split} valid={len(rows)-split}")


if __name__ == "__main__":
    main()
