import numpy as np
import onnxruntime as ort
import torch

from training.token_export_onnx import export_model
from training.token_model import OUTPUT_HEADS, TokenAttributeModel


def test_exported_token_model_has_stable_io_and_matches_pytorch(tmp_path):
    path = tmp_path / "token_ai.onnx"
    model = TokenAttributeModel(embedding_dim=8, hidden_dim=8)
    model.eval()
    input_ids = torch.zeros((1, 32), dtype=torch.long)
    export_model(model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["input_ids"]
    assert [item.name for item in session.get_outputs()] == list(OUTPUT_HEADS)
    outputs = session.run(None, {"input_ids": input_ids.numpy()})
    assert outputs[0].shape[1] == model.label_count
    assert outputs[1].shape == (1, model.label_count, 5)

    with torch.no_grad():
        expected = model(input_ids)
    for actual, name in zip(outputs, OUTPUT_HEADS):
        np.testing.assert_allclose(actual, expected[name].numpy(), rtol=1e-4, atol=1e-5)
