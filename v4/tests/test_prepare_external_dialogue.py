import json

from scripts.prepare_external_dialogue import (
    clean_text,
    load_chatbot_rows,
    load_empathetic_rows,
    make_sft_rows,
)


def test_training_validation_directories_are_separate(tmp_path):
    from scripts.train_hrm import build_train_val_datasets

    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    (train_dir / "data.txt").write_text("Q: 첫 질문\nA: 첫 답\n\nQ: 둘째 질문\nA: 둘째 답\n", encoding="utf-8")
    (valid_dir / "data.txt").write_text("Q: 검증 질문\nA: 검증 답\n", encoding="utf-8")

    train, valid = build_train_val_datasets(str(train_dir), str(valid_dir))
    assert len(train) == 2
    assert len(valid) == 1


def test_clean_text_removes_control_noise_and_keeps_korean_sentence():
    assert clean_text("  오늘\t기분은\n좋아!  ") == "오늘 기분은 좋아!"
    assert clean_text("http://example.com 광고") == ""


def test_loaders_and_formatter_produce_q_a_rows(tmp_path):
    emp = tmp_path / "emp.jsonl"
    emp.write_text(json.dumps({
        "instruction": "오늘 너무 힘들어요.",
        "output": "많이 힘드셨겠어요. 괜찮다면 이야기해 주세요.",
        "type": "single",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    chatbot = tmp_path / "chat.csv"
    chatbot.write_text("Q,A,label\n안녕,반가워요.,0\n", encoding="utf-8")

    rows = load_empathetic_rows(emp) + load_chatbot_rows(chatbot)
    blocks = make_sft_rows(rows, max_items=10)

    assert len(blocks) == 2
    assert blocks[0].startswith("Q: ")
    assert "\nA: " in blocks[0]
