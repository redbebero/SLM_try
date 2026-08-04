import sys

sys.path.insert(0, "scripts")

from dialogue_hybrid import DialogueManager


def test_manager_uses_multiple_candidates_for_emotion_dialogue():
    manager = DialogueManager()
    candidates = manager.candidates("오늘 하루 종일 마음이 무겁고 지쳤어.")
    assert len(candidates) == 2
    assert any("무슨 일이" in item or "힘든" in item for item in candidates)


def test_manager_remembers_activity_across_new_wording():
    manager = DialogueManager()
    assert "기억" in manager.reply("나는 주말마다 등산해")
    answer = manager.reply("내가 주말에 뭘 한다고 했지?")
    assert "등산" in answer


def test_manager_routes_known_fact_to_specialist():
    manager = DialogueManager()
    answer = manager.reply("한글을 만든 왕은 누구야?")
    assert "세종" in answer
