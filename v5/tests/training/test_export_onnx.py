import numpy as np
import onnxruntime as ort
import torch

from training.export_onnx import export_model
from training.model import OUTPUT_HEADS, SpellProposalModel


def test_exported_model_has_stable_godot_facing_io(tmp_path):
    path = tmp_path / "spell_ai.onnx"
    model = SpellProposalModel(embedding_dim=8, hidden_dim=8)
    model.eval()
    input_ids = torch.zeros((1, 96), dtype=torch.long)
    export_model(model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["input_ids"]
    assert [item.name for item in session.get_outputs()] == list(OUTPUT_HEADS)

    outputs = session.run(None, {"input_ids": input_ids.numpy()})
    assert len(outputs) == len(OUTPUT_HEADS)
    assert all(output.shape[0] == 1 for output in outputs)

    with torch.no_grad():
        torch_outputs = model(input_ids)
    for actual, name in zip(outputs, OUTPUT_HEADS):
        np.testing.assert_allclose(actual, torch_outputs[name].numpy(), rtol=1e-4, atol=1e-5)
