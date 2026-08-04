"""Audit one downloaded SFT record's target shift, mask, and EOS."""
import argparse
import json
from pathlib import Path

from dataset import KoJamoDataset


def resolve_data_dir(path):
    """Accept either an SFT split directory or its dataset root."""
    path = Path(path)
    if (path / "records.jsonl").exists():
        return path
    train = path / "train"
    if (train / "records.jsonl").exists():
        return train
    raise FileNotFoundError(
        f"SFT records.jsonl not found in {path} or {train}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    root = resolve_data_dir(args.data_dir)
    ds = KoJamoDataset(data_dir=str(root), is_sft=True)
    x, y, mask = ds[0]
    tok = ds.tokenizer
    raw = json.loads((root / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    full = tok.decode(torch_cat(x, y[-1:]))
    prompt_len = int((mask == 0).sum().item())
    active = mask.nonzero(as_tuple=False).flatten()
    first = int(active[0]) if len(active) else -1
    eos_id = tok.sym_vocab["\n"]
    result = {
        "rows": len(ds),
        "question": raw["question"],
        "answer": raw["answer"],
        "decoded_full": full,
        "prompt_token_count": prompt_len,
        "first_active_target_index": first,
        "first_active_target_decoded": tok.decode(y[first:first + 1]) if first >= 0 else "",
        "last_target_decoded": tok.decode(y[-1:]),
        "eos_id": eos_id,
        "last_target_is_eos": bool(y[-1, 3].item() == eos_id and y[-1, 0:3].sum().item() == 0),
        "x_at_first_active_is_prompt_token": bool(first < len(x) and x[first].sum().item() > 0) if first >= 0 else False,
        "prompt_mask_zero": bool(mask[:first].sum().item() == 0) if first >= 0 else False,
        "answer_mask_active": bool(mask[first:].sum().item() > 0) if first >= 0 else False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def torch_cat(x, tail):
    import torch
    return torch.cat([x, tail], dim=0)


if __name__ == "__main__":
    main()
