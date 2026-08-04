"""Export the trained proposal model with stable tensor names."""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from .model import OUTPUT_HEADS, SpellProposalModel


class OnnxOutputWrapper(nn.Module):
    def __init__(self, model: SpellProposalModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor) -> tuple[Tensor, ...]:
        outputs = self.model(input_ids)
        return tuple(outputs[name] for name in OUTPUT_HEADS)


def export_model(model: SpellProposalModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = OnnxOutputWrapper(model)
    dummy_input = torch.zeros((1, 96), dtype=torch.long)
    dynamic_axes = {
        "input_ids": {0: "batch"},
        **{name: {0: "batch"} for name in OUTPUT_HEADS},
    }
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        destination,
        input_names=["input_ids"],
        output_names=list(OUTPUT_HEADS),
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/spell_ai.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/spell_ai.onnx"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SpellProposalModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    export_model(model, args.output)
    print({"onnx": str(args.output), "outputs": OUTPUT_HEADS})


if __name__ == "__main__":
    main()
