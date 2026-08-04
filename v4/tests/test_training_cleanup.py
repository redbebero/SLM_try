import sys

import torch

sys.path.insert(0, "scripts")

from dataset import KoJamoDataset
from dataset import sft_text_with_eos
from model import KoJamoNet, KoJamoTransformer
from train import effective_warmup_steps, training_batch_limit
from chat import infer_checkpoint_spec, sample_logits, get_context_length, select_korean_jamo, resolve_checkpoint
from tokenizer import KoJamoTokenizer
from prepare_clean_pretrain import accept
from prepare_hrm_reasoning import make_tasks, task_type
from train_sft import build_sft_model, sft_collate_fn, apply_sft_input_dropout
from train_hrm import build_scheduled_input
from hrm_model import HRMJamoNet
from hrm_model import HRMContextNet
from reasoning_router import try_reasoning_answer
from evaluate_generation import repetition_3gram_ratio, score_generation
from audit_sft_data import audit_blocks, normalize_answer
from audit_sft_alignment import resolve_data_dir


def test_tokenizer_round_trip_for_hangul_and_tracks():
    tokenizer = KoJamoTokenizer()
    text = "가각힣ㄱㅏㄳ A9!?\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_alignment_audit_resolves_dataset_root_to_train_split(tmp_path):
    root = tmp_path / "downloaded"
    train = root / "train"
    train.mkdir(parents=True)
    (train / "records.jsonl").write_text(
        '{"question":"질문","answer":"답변"}\n', encoding="utf-8"
    )

    assert resolve_data_dir(root) == train


def test_alignment_audit_keeps_explicit_split_directory(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "records.jsonl").write_text(
        '{"question":"질문","answer":"답변"}\n', encoding="utf-8"
    )

    assert resolve_data_dir(split) == split


def test_pretraining_windows_stay_inside_source_samples(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "samples.txt").write_text("가나다라마바사\n아자차카타파하\n", encoding="utf-8")

    dataset = KoJamoDataset(data_dir=str(data_dir), seq_length=4, stride=1)

    assert hasattr(dataset, "sample_starts")
    assert len(dataset.sample_starts) == len(dataset)
    for index in range(len(dataset)):
        x, y = dataset[index]
        assert x.shape == (4, 6)
        assert y.shape == (4, 6)
        assert "\n" not in dataset.tokenizer.decode(x)
    assert any("\n" in dataset.tokenizer.decode(dataset[index][1])
               for index in range(len(dataset)))


def test_pretraining_cache_rebuilds_when_window_length_changes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "samples.txt").write_text("가나다라마바사\n아자차카타파하\n", encoding="utf-8")

    short = KoJamoDataset(data_dir=str(data_dir), seq_length=4, stride=1)
    long = KoJamoDataset(data_dir=str(data_dir), seq_length=6, stride=1)

    assert short[0][0].shape == (4, 6)
    assert long[0][0].shape == (6, 6)


def test_finite_training_warmup_scales_to_run_length():
    assert effective_warmup_steps(total_steps=150, configured_steps=300) == 7
    assert effective_warmup_steps(total_steps=6000, configured_steps=300) == 300


def test_cleaner_rejects_unknown_and_reference_text():
    assert not accept("가" * 100 + "漢")
    assert not accept("공식 홈페이지에 대한 설명" + "가" * 50)


def test_model_uses_previous_jong_embedding_once():
    model = KoJamoNet(vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=64, hidden_dim=128, num_layers=1)
    assert model.proj_in.in_features == 64 * 6 + 32


def test_model_can_use_independent_jamo_heads():
    model = KoJamoNet(
        vocab_sizes=(20, 22, 29, 36, 53, 11),
        emb_dim=32,
        hidden_dim=64,
        num_layers=1,
        cascade=False,
    )
    x = torch.zeros(2, 8, 6, dtype=torch.long)
    outputs = model(x)
    assert [out.shape[-1] for out in outputs] == [4, 20, 22, 29, 36, 53, 11]
    assert model.head_jung.in_features == 64
    assert model.head_jong.in_features == 64


def test_transformer_has_causal_jamo_fusion_and_expected_outputs():
    model = KoJamoTransformer(
        vocab_sizes=(20, 22, 29, 36, 53, 11),
        emb_dim=32,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
    )
    x = torch.zeros(2, 8, 6, dtype=torch.long)
    outputs = model(x)
    assert [out.shape[-1] for out in outputs] == [4, 20, 22, 29, 36, 53, 11]
    assert model.fusion_proj.out_features == 64


def test_limited_training_length_is_respected():
    assert training_batch_limit(100, 7) == 7
    assert training_batch_limit(5, 7) == 5
    assert training_batch_limit(100, None) == 100


def test_chat_infers_transformer_checkpoint_shape():
    state = {
        "pos_emb.weight": torch.zeros(512, 384),
        "emb_cho.weight": torch.zeros(20, 48),
        "head_type.weight": torch.zeros(4, 384),
        "core.layers.0.self_attn.in_proj_weight": torch.zeros(1152, 384),
        "core.layers.3.self_attn.in_proj_weight": torch.zeros(1152, 384),
    }
    spec = infer_checkpoint_spec(state)
    assert spec == {
        "variant": "transformer",
        "emb_dim": 48,
        "hidden_dim": 384,
        "num_layers": 4,
        "num_heads": 6,
        "max_seq_length": 512,
    }


def test_chat_infers_all_gru_layers_from_state_dict():
    hidden = 128
    state = {
        "emb_cho.weight": torch.zeros(20, 16),
        "head_type.weight": torch.zeros(4, hidden),
        "head_jung.weight": torch.zeros(22, hidden),
        "head_jong.weight": torch.zeros(28, hidden),
        "core.weight_ih_l0": torch.zeros(3 * hidden, hidden),
        "core.weight_ih_l1": torch.zeros(3 * hidden, hidden),
        "core.weight_hh_l0": torch.zeros(3 * hidden, hidden),
        "core.weight_hh_l1": torch.zeros(3 * hidden, hidden),
    }
    spec = infer_checkpoint_spec(state)
    assert spec["num_layers"] == 2


def test_sample_logits_applies_repetition_penalty_and_top_k():
    log_probs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), dim=-1)
    previous = torch.tensor([[3]])
    sampled = sample_logits(
        log_probs,
        temperature=1.0,
        top_k=2,
        repetition_penalty=2.0,
        previous_ids=previous,
    )
    assert sampled.shape == (1,)
    assert sampled.item() in (2, 3)


def test_sample_logits_temperature_zero_is_greedy():
    log_probs = torch.log_softmax(torch.tensor([[1.0, 4.0, 2.0]]), dim=-1)
    sampled = sample_logits(log_probs, temperature=0.0)
    assert sampled.tolist() == [1]


def test_sft_builder_loads_transformer_checkpoint():
    tokenizer = KoJamoTokenizer()
    model = build_sft_model(
        tokenizer.get_vocab_sizes(),
        checkpoint_path="checkpoints/unattended_full_best.pth",
        device=torch.device("cpu"),
    )
    assert isinstance(model, KoJamoTransformer)


def test_sft_collate_truncates_to_transformer_context():
    x = torch.zeros(600, 6, dtype=torch.long)
    y = torch.zeros(600, 6, dtype=torch.long)
    mask = torch.ones(600)
    batch_x, batch_y, batch_mask = sft_collate_fn([(x, y, mask)], max_seq_length=512)
    assert batch_x.shape == (1, 512, 6)
    assert batch_y.shape == (1, 512, 6)
    assert batch_mask.shape == (1, 512)


def test_generation_metrics_detect_repeated_ngrams():
    assert repetition_3gram_ratio("가나다라마바사 가나다라마바사 가나다라마바사 가나다라마바사 가나다라마바사") > 0


def test_generation_score_counts_expected_keywords():
    score = score_generation("서울은 대한민국의 수도입니다", ["서울", "수도"])
    assert score["keyword_hit_rate"] == 1.0


def test_transformer_context_length_comes_from_position_embedding():
    model = KoJamoTransformer(
        vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=32,
        hidden_dim=64, num_layers=1, num_heads=4, max_seq_length=512,
    )
    assert get_context_length(model) == 512


def test_sft_audit_rejects_malformed_and_duplicate_blocks():
    blocks = [
        "Q: 수도는?\nA: 서울입니다.",
        "Q: 수도는?\nA: 서울입니다.",
        "깨진 블록",
        "Q: 빈 답\nA: ",
    ]
    kept, stats = audit_blocks(blocks)
    assert kept == ["Q: 수도는?\nA: 서울입니다."]
    assert stats["duplicate"] == 1
    assert stats["malformed"] == 2


def test_sft_audit_rejects_refusal_template_answers():
    blocks = [
        "Q: 누가 만들었나요?\nA: 저는 배경지식이 없어 알지 못합니다.",
        "Q: 수도는?\nA: 서울입니다.",
    ]
    kept, stats = audit_blocks(blocks)
    assert kept == ["Q: 수도는?\nA: 서울입니다."]
    assert stats["low_quality"] == 1


def test_sft_normalizes_only_explicit_final_answer_patterns():
    assert normalize_answer("질문", '근거를 보면 질문의 답은 "채리티 쇼"입니다.') == "채리티 쇼입니다."
    assert normalize_answer("계산", "풀이 과정입니다.\n#### 42") == "정답은 42입니다."
    assert normalize_answer("대화", "안녕하세요. 무엇을 도와드릴까요?") == "안녕하세요. 무엇을 도와드릴까요?"


def test_sft_target_contains_explicit_end_of_answer_newline():
    assert sft_text_with_eos("Q: 질문\nA: 답변").endswith("답변\n")


def test_sft_input_dropout_only_masks_answer_positions():
    x = torch.ones(1, 4, 6, dtype=torch.long)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    dropped = apply_sft_input_dropout(x, mask, probability=1.0)
    assert torch.equal(dropped[:, :2], x[:, :2])
    assert torch.equal(dropped[:, 2:], torch.zeros(1, 2, 6, dtype=torch.long))


def test_korean_decoder_penalizes_repeated_whole_syllables():
    cho = torch.log_softmax(torch.tensor([[0.0, 5.0, 1.0]]), dim=-1)
    jung = torch.log_softmax(torch.tensor([[0.0, 5.0, 1.0]]), dim=-1)
    jong = torch.log_softmax(torch.tensor([[5.0, 1.0]]), dim=-1)
    result = select_korean_jamo(cho, jung, jong, recent_syllables={(1, 1, 0)}, repetition_penalty=100.0)
    assert tuple(item.item() for item in result) != (1, 1, 0)


def test_chat_defaults_to_validated_full_pretrain_checkpoint():
    assert resolve_checkpoint() == "checkpoints/unattended_full_best.pth"


def test_hrm_lite_has_two_timescale_state_and_jamo_outputs():
    model = HRMJamoNet((20, 22, 29, 36, 53, 11), emb_dim=8, hidden_dim=16, cycle_steps=2)
    x = torch.zeros(2, 6, 6, dtype=torch.long)
    logits, state = model(x)
    assert len(logits) == 7
    assert all(item.shape[:2] == (2, 6) for item in logits)
    assert state[0].shape == state[1].shape == (2, 16)


def test_hrm_attention_variant_keeps_hierarchical_outputs():
    model = HRMJamoNet((20, 22, 29, 36, 53, 11), emb_dim=8, hidden_dim=32,
                       cycle_steps=2, use_attention=True)
    x = torch.zeros(2, 6, 6, dtype=torch.long)
    logits, _ = model(x)
    assert model.use_attention
    assert logits[0].shape == (2, 6, 4)


def test_hrm_prompt_memory_accepts_prompt_mask():
    model = HRMJamoNet((20, 22, 29, 36, 53, 11), emb_dim=8, hidden_dim=32,
                       cycle_steps=2, use_prompt_memory=True)
    x = torch.zeros(2, 8, 6, dtype=torch.long)
    prompt_mask = torch.tensor([[True] * 4 + [False] * 4, [True] * 3 + [False] * 5])
    logits, _ = model(x, prompt_mask=prompt_mask)
    assert model.use_prompt_memory
    assert logits[0].shape == (2, 8, 4)


def test_hrm_context_net_preserves_causality_and_hierarchical_outputs():
    model = HRMContextNet((20, 22, 29, 36, 53, 11), emb_dim=8,
                          hidden_dim=32, context_layers=1, cycle_steps=2)
    x = torch.zeros(2, 10, 6, dtype=torch.long)
    x[:, :, 0] = torch.randint(0, 20, (2, 10))
    logits, state = model(x, segments=2)
    assert len(logits) == 7
    assert logits[0].shape == (2, 10, 4)
    assert state[0].shape == (2, 32)
    assert hasattr(model, "jamo_fusion")
    # A causal model must not expose future input changes at earlier positions.
    x2 = x.clone()
    x2[:, 7:, 0] = (x2[:, 7:, 0] + 3) % 20
    first, _ = model(x, segments=1)
    second, _ = model(x2, segments=1)
    assert torch.allclose(first[0][:, :7], second[0][:, :7], atol=1e-5)


def test_hrm_context_copy_head_is_optional_and_causal():
    model = HRMContextNet((20, 22, 29, 36, 53, 11), emb_dim=8,
                          hidden_dim=32, context_layers=1, cycle_steps=2,
                          use_copy=True)
    x = torch.zeros(1, 8, 6, dtype=torch.long)
    x[0, 0, 0] = 3
    x[0, 0, 1] = 4
    logits, _ = model(x)
    assert model.use_copy
    assert all(item.shape[1:] == (8, size) for item, size in zip(
        logits, (4, 20, 22, 29, 36, 53, 11)))


def test_hrm_context_can_read_current_final_consonant():
    model = HRMContextNet((20, 22, 29, 36, 53, 11), emb_dim=8,
                          hidden_dim=32, context_layers=1, cycle_steps=2,
                          use_current_jong=True)
    x = torch.zeros(1, 8, 6, dtype=torch.long)
    x[0, 4, 2] = 1
    first, _ = model(x)
    x[0, 4, 2] = 7
    second, _ = model(x)
    assert model.use_current_jong
    assert not torch.allclose(first[0][:, 4], second[0][:, 4])
    assert torch.allclose(first[0][:, :4], second[0][:, :4], atol=1e-5)


def test_hrm_joint_jamo_head_scores_valid_syllable_combinations():
    model = HRMContextNet((20, 22, 29, 36, 53, 11), emb_dim=8,
                          hidden_dim=32, context_layers=1, cycle_steps=2,
                          use_joint_jamo=True)
    x = torch.zeros(1, 6, 6, dtype=torch.long)
    logits, _ = model(x)
    assert model.use_joint_jamo
    assert model.last_joint_logits.shape == (1, 6, 20, 22, 29)
    assert len(logits) == 7


def test_reasoning_router_solves_structured_jamo_tasks_without_hijacking_chat():
    assert try_reasoning_answer(
        "Q: [산수] 상자에 12개가 있습니다. 7개를 더 넣고 4개를 꺼냈습니다. 남은 개수는?"
    ) == "12+7-4=15이므로 정답은 15개입니다."
    assert try_reasoning_answer(
        "Q: [자소] 초성 ㄱ, 중성 ㅏ, 종성 ㄴ을 합치면 어떤 글자인가요?"
    ) == "ㄱ, ㅏ, ㄴ을 합치면 간이므로 정답은 간입니다."
    assert try_reasoning_answer(
        "Q: [순서] 다음 계절을 빠른 순서대로 배열하세요: 겨울 봄 가을"
    ) == "계절의 순서는 봄 가을 겨울입니다."
    assert try_reasoning_answer("3 곱하기 4는?") == "3×4=12입니다."
    assert try_reasoning_answer("초성 ㄱ, 중성 ㅏ, 종성 ㄴ을 합치면?") == "ㄱ, ㅏ, ㄴ을 합치면 간이므로 정답은 간입니다."
    assert try_reasoning_answer("계절을 순서대로 말해줘: 겨울 봄 가을") == "계절의 순서는 봄 가을 겨울입니다."
    assert try_reasoning_answer("나는 부산에 살고 있어. 내가 사는 도시는") == "부산입니다."
    assert try_reasoning_answer("대한민국의 수도는 어디인가요?") == "서울입니다."
    assert "기분은 괜찮아요" in try_reasoning_answer("오늘 기분이 어때?")


def test_conversation_state_recalls_activity_with_ta_conjugation():
    prompt = "Q: 나는 주말마다 자전거를 타\nA: 말해준 취미와 활동을 기억해둘게요.\nQ: 내가 주말마다 뭘 한다고 했지?\nA: "
    assert "자전거" in try_reasoning_answer(prompt)


def test_downloaded_sft_deduplicates_source_pairs_and_preserves_metadata():
    from build_downloaded_sft import deduplicate, make_record

    raw = {"messages": [{"role": "user", "content": "질문"}]}
    first = make_record("질문", "답변", "quality:test", "1", "dialogue", raw)
    duplicate = make_record("질문", "답변", "quality:test", "2", "dialogue", raw)
    records = deduplicate([first, duplicate])
    assert len(records) == 1
    assert records[0]["source"] == "quality:test"
    assert records[0]["raw_hash"]
    assert records[0]["pair_hash"]


def test_chat_infers_hrm_checkpoint_shape():
    state = {
        "emb_cho.weight": torch.zeros(20, 16),
        "l_cell.weight_ih": torch.zeros(384, 256),
        "l_cell.weight_hh": torch.zeros(384, 128),
        "h_cell.weight_ih": torch.zeros(384, 128),
    }
    spec = infer_checkpoint_spec(state)
    assert spec["variant"] == "hrm"
    assert spec["emb_dim"] == 16
    assert spec["hidden_dim"] == 128


def test_hrm_reasoning_generator_is_balanced_varied_and_deterministic():
    tasks_a = make_tasks(90, seed=19)
    tasks_b = make_tasks(90, seed=19)

    assert tasks_a == tasks_b
    assert len(tasks_a) == 90
    assert len(set(tasks_a)) == 90
    assert {kind: sum(task_type(task) == kind for task in tasks_a)
            for kind in ("arithmetic", "jamo", "ordering")} == {
                "arithmetic": 30, "jamo": 30, "ordering": 30,
            }
    assert len({task.split("\nA: ", 1)[0] for task in tasks_a
                if task_type(task) == "arithmetic"}) >= 20
    assert all("정답은 " in task.split("\nA: ", 1)[1]
               for task in tasks_a if task_type(task) != "ordering")


def test_hrm_scheduled_input_preserves_prompt_and_replaces_answer_history():
    x = torch.tensor([[[1, 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0]]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    logits = tuple(torch.zeros(1, 3, size) for size in (4, 20, 22, 29, 36, 53, 11))
    logits = tuple(item + torch.arange(item.size(-1), dtype=torch.float).view(1, 1, -1) for item in logits)
    result = build_scheduled_input(x, logits, mask, probability=1.0)
    assert torch.equal(result[:, 0], x[:, 0])
    assert torch.equal(result[:, 1], x[:, 1])
    assert not torch.equal(result[:, 2], x[:, 2])
