"""Build deterministic subject-conditioned toy data for compositional evaluation."""

from pathlib import Path


SUBJECTS = (
    ("나는", "먹는다"),
    ("너는", "본다"),
    ("어머니가", "마신다"),
    ("아버지가", "읽는다"),
    ("강아지가", "산다"),
    ("고양이가", "던진다"),
    ("학생이", "좋아한다"),
    ("우리는", "싫어한다"),
)
OBJECTS = ("밥", "책", "물", "공")


def build_samples():
    train, valid = [], []
    for subject_index, (subject, verb) in enumerate(SUBJECTS):
        for object_index, obj in enumerate(OBJECTS):
            line = f"{subject} {obj}을 {verb}."
            # Hold out one object per subject. Every subject and object remains
            # visible in train, but this exact pair is unseen in validation.
            if object_index == subject_index % len(OBJECTS):
                valid.append(line)
            else:
                train.append(line)
    return train, valid


def write_dataset(root="experiments/compositional_toy"):
    root = Path(root)
    train, valid = build_samples()
    train_dir = root / "train"
    valid_dir = root / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (valid_dir / "valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    return train, valid


if __name__ == "__main__":
    train, valid = write_dataset()
    print(f"train={len(train)} valid={len(valid)}")
