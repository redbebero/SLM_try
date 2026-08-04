import argparse
import io
import json
import sys

import pytest
import torch

from training.token_cli import _threshold, classify_sentence, classify_tokens, main, render_terminal_output
from training.token_model import TokenAttributeModel
from training.token_train import save_checkpoint


class FixedModel:
    def __call__(self, input_ids):
        batch_size = input_ids.shape[0]
        attribute_logits = torch.full((batch_size, 2), -5.0)
        attribute_logits[:, 0] = 5.0
        delta_logits = torch.zeros((batch_size, 2, 5))
        delta_logits[:, 0, 3] = 5.0
        return {
            "attribute_logits": attribute_logits,
            "delta_logits": delta_logits,
        }


def test_classify_tokens_preserves_whitespace_token_boundaries():
    results = classify_tokens(
        FixedModel(),
        ["ELEMENT:FIRE", "FORM:ORB"],
        ["붉은", "구체"],
        max_length=32,
    )

    assert [result["token"] for result in results] == ["붉은", "구체"]
    assert results[0]["attributes"][0]["value"] == "FIRE"
    assert results[1]["attributes"][0]["value"] == "FIRE"


def test_classify_sentence_splits_only_on_whitespace():
    results = classify_sentence(
        "붉은 구체",
        FixedModel(),
        ["ELEMENT:FIRE", "FORM:ORB"],
        max_length=32,
    )

    assert [result["token"] for result in results] == ["붉은", "구체"]


def test_render_terminal_output_is_readable():
    output = render_terminal_output(
        "붉은 구체",
        [
            {
                "schema_version": 1,
                "token": "붉은",
                "attributes": [
                    {"kind": "ELEMENT", "value": "FIRE", "delta": 1, "confidence": 0.9876},
                ],
                "unknown": False,
            },
        ],
    )

    assert "INPUT: 붉은 구체" in output
    assert "붉은" in output
    assert "ELEMENT:FIRE" in output
    assert "delta=+1" in output
    assert "confidence=0.988" in output


def test_main_reads_stdin_and_prints_json(tmp_path, monkeypatch, capsys):
    checkpoint_path = tmp_path / "token_ai.pt"
    save_checkpoint(TokenAttributeModel(), checkpoint_path)
    monkeypatch.setattr(sys, "argv", ["token_cli", "--model", str(checkpoint_path), "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("붉은 구체\n"))

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input"] == "붉은 구체"
    assert [row["token"] for row in payload["tokens"]] == ["붉은", "구체"]


def test_threshold_rejects_values_outside_probability_range():
    assert _threshold("0.8") == 0.8
    with pytest.raises(argparse.ArgumentTypeError, match="between 0 and 1"):
        _threshold("1.1")
