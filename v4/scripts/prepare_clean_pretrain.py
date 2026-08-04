"""Build a Korean-focused pretraining corpus from train_data/*.txt.

Output is split into separate train/validation directories because KoJamoDataset
loads every .txt file in a directory.
"""

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "train_data"
TRAIN_DIR = ROOT / "train_data_clean"
VAL_DIR = ROOT / "val_data_clean"
VAL_EVERY = 100  # deterministic 1% validation split
MAX_TRAIN_CHARS = 120_000_000  # keep token-cache creation within laptop RAM limits

KO_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
LATIN_RE = re.compile(r"[A-Za-z]")
HANJA_RE = re.compile(r"[一-鿿]")
DIGIT_RE = re.compile(r"[0-9]")
UNKNOWN_RE = re.compile(r"[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9 !\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~\n\r\t]")
REFERENCE_RE = re.compile(
    r"(isbn|doi\s*:|retrieved from|references|bibliography|공식\s*홈페이지|저널|vol\.\s*\d+|pp\.\s*\d+)",
    re.IGNORECASE,
)


def accept(text: str) -> bool:
    text = text.strip()
    if not 50 <= len(text) <= 800:
        return False
    ko = len(KO_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    hanja = len(HANJA_RE.findall(text))
    letters = ko + latin + hanja
    if letters == 0 or ko / letters < 0.70:
        return False
    # Unsupported characters become the tokenizer's UNK symbol. One such
    # character is enough to make a training sample unreliable.
    if UNKNOWN_RE.search(text):
        return False
    if REFERENCE_RE.search(text):
        return False
    return True


def main() -> None:
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    train_path = TRAIN_DIR / "pretrain_train.txt"
    val_path = VAL_DIR / "pretrain_val.txt"

    seen = set()
    train_chars = val_chars = train_count = val_count = rejected = 0
    with train_path.open("w", encoding="utf-8") as train_out, val_path.open("w", encoding="utf-8") as val_out:
        for source in sorted(SOURCE_DIR.glob("*.txt")):
            with source.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    text = " ".join(raw.strip().split())
                    if not accept(text):
                        rejected += 1
                        continue
                    digest = hashlib.sha1(text.encode("utf-8")).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    if len(seen) % VAL_EVERY == 0:
                        val_out.write(text + "\n")
                        val_count += 1
                        val_chars += len(text)
                    else:
                        train_out.write(text + "\n")
                        train_count += 1
                        train_chars += len(text)
                        if train_chars >= MAX_TRAIN_CHARS:
                            break
                if train_chars >= MAX_TRAIN_CHARS:
                    break

    print(f"train: {train_count:,} lines / {train_chars:,} chars -> {train_path}")
    print(f"valid: {val_count:,} lines / {val_chars:,} chars -> {val_path}")
    print(f"rejected: {rejected:,} lines")


if __name__ == "__main__":
    main()
