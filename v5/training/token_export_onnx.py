"""Export token attribute model with stable Godot-facing tensor names."""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from .token_model import OUTPUT_HEADS, TOKEN_MAX_LENGTH, TokenAttributeModel


class OnnxTokenOutputWrapper(nn.Module):
    def __init__(self, model: TokenAttributeModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(input_ids)
        return tuple(outputs[name] for name in OUTPUT_HEADS)


def export_model(model: TokenAttributeModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        OnnxTokenOutputWrapper(model),
        (torch.zeros((1, TOKEN_MAX_LENGTH), dtype=torch.long),),
        destination,
        input_names=["input_ids"],
        output_names=list(OUTPUT_HEADS),
        dynamic_axes={"input_ids": {0: "batch"}, **{name: {0: "batch"} for name in OUTPUT_HEADS}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/token_ai.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/token_ai.onnx"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TokenAttributeModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    export_model(model, args.output)
    print({"onnx": str(args.output), "outputs": OUTPUT_HEADS, "labels": len(checkpoint["labels"])})


if __name__ == "__main__":
    main()
