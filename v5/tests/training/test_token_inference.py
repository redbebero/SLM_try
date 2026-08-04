import torch

from training.token_inference import logits_to_token_classification


def test_inference_decodes_multiple_attributes_and_delta():
    attribute_logits = torch.full((1, 2), -4.0)
    attribute_logits[0, 0] = 4.0
    attribute_logits[0, 1] = 4.0
    delta_logits = torch.zeros((1, 2, 5))
    delta_logits[0, 0, 3] = 4.0  # +1
    delta_logits[0, 1, 2] = 4.0  # 0

    result = logits_to_token_classification(
        {"attribute_logits": attribute_logits, "delta_logits": delta_logits},
        ["ELEMENT:FIRE", "FORM:ORB"],
        token="붉은 구체",
        threshold=0.5,
    )[0]

    assert result["unknown"] is False
    assert result["attributes"] == [
        {"kind": "ELEMENT", "value": "FIRE", "delta": 1, "confidence": result["attributes"][0]["confidence"]},
        {"kind": "FORM", "value": "ORB", "delta": 0, "confidence": result["attributes"][1]["confidence"]},
    ]


def test_inference_marks_no_matching_attribute_unknown():
    result = logits_to_token_classification(
        {"attribute_logits": torch.full((1, 1), -4.0), "delta_logits": torch.zeros((1, 1, 5))},
        ["ELEMENT:FIRE"],
        token="평범한",
    )[0]

    assert result["attributes"] == []
    assert result["unknown"] is True
