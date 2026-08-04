"""Terminal interface for Korean one-token magic attribute classification."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .encoding import encode_incantation
from .token_inference import logits_to_token_classification
from .token_model import TokenAttributeModel


def load_token_checkpoint(path: str | Path) -> tuple[TokenAttributeModel, list[str], int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TokenAttributeModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["labels"], checkpoint["model_config"]["max_length"]


def classify_tokens(
    model: Any,
    labels: Sequence[str],
    tokens: Sequence[str],
    *,
    max_length: int,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    if not tokens:
        return []
    input_ids = torch.tensor(
        [encode_incantation(token, max_length) for token in tokens],
        dtype=torch.long,
    )
    with torch.no_grad():
        logits = model(input_ids)
    results = logits_to_token_classification(
        logits,
        labels,
        token="",
        threshold=threshold,
    )
    for token, result in zip(tokens, results):
        result["token"] = token
    return results


def classify_sentence(
    sentence: str,
    model: Any,
    labels: Sequence[str],
    *,
    max_length: int,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    return classify_tokens(
        model,
        labels,
        sentence.split(),
        max_length=max_length,
        threshold=threshold,
    )


def render_terminal_output(sentence: str, results: Sequence[dict[str, Any]]) -> str:
    lines = [f"INPUT: {sentence}"]
    for index, result in enumerate(results, start=1):
        token = result["token"]
        if result["unknown"]:
            lines.append(f"[{index:02d}] {token} -> UNKNOWN")
            continue
        lines.append(f"[{index:02d}] {token}")
        for attribute in result["attributes"]:
            lines.append(
                "  - "
                f"{attribute['kind']}:{attribute['value']} "
                f"delta={attribute['delta']:+d} "
                f"confidence={attribute['confidence']:.3f}"
            )
    return "\n".join(lines)


def _threshold(value: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Korean incantation; stdin is used when omitted")
    parser.add_argument("--model", type=Path, default=Path("models/token_ai.pt"))
    parser.add_argument("--threshold", type=_threshold, default=0.5)
    parser.add_argument("--json", action="store_true", dest="as_json", help="print machine-readable JSON")
    args = parser.parse_args()

    sentence = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
    if not sentence:
        parser.error("Korean incantation is required as an argument or stdin")

    model, labels, max_length = load_token_checkpoint(args.model)
    results = classify_sentence(
        sentence,
        model,
        labels,
        max_length=max_length,
        threshold=args.threshold,
    )
    if args.as_json:
        print(json.dumps({"input": sentence, "tokens": results}, ensure_ascii=False, indent=2))
    else:
        print(render_terminal_output(sentence, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
