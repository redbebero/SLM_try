import torch
import sys
import os
import glob
import re
import argparse

# v2 폴더 기준 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet, KoJamoTransformer
from hrm_model import HRMJamoNet, HRMContextNet, HRMConditionalNet
from reasoning_router import try_reasoning_answer
from hangul_semantic_plan import analyze_question, render_plan
from dialogue_hybrid import DialogueManager
from context_extractor import extract_passage_answer
from dialogue_intent import load_intent_checkpoint, predict_intent
from knowledge_memory import KnowledgeMemory
from tokenizer import KoJamoTokenizer
from config import EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, CHECKPOINT_PREFIX


def infer_checkpoint_spec(state_dict):
    """Infer the model family and dimensions from a saved state dict."""
    if "l_cell.weight_ih" in state_dict and "h_cell.weight_ih" in state_dict:
        context_layers = [
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"context_encoder\.layers\.(\d+)\.", key))
        ]
        return {
            "variant": "hrm_context" if context_layers else "hrm",
            "emb_dim": state_dict["emb_cho.weight"].shape[1],
            "hidden_dim": state_dict["l_cell.weight_hh"].shape[1],
            "num_layers": None,
            "num_heads": None,
            "max_seq_length": state_dict.get("pos_emb.weight", torch.empty(250, 1)).shape[0],
            "cycle_steps": 4,
            "use_attention": "context_attn.in_proj_weight" in state_dict,
            "use_prompt_memory": "prompt_attn.in_proj_weight" in state_dict,
            "context_layers": max(context_layers, default=0) + 1,
            "use_copy": "copy_gate.weight" in state_dict,
            "use_current_jong": "current_jong_proj.weight" in state_dict,
            "use_joint_jamo": "joint_query.weight" in state_dict,
            "use_query_summary": "query_summary_proj.weight" in state_dict,
            "use_char_head": "char_head.weight" in state_dict,
        }
    if "pos_emb.weight" in state_dict:
        layer_numbers = [
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"core\.layers\.(\d+)\.", key))
        ]
        hidden_dim = state_dict["head_type.weight"].shape[1]
        return {
            "variant": "transformer",
            "emb_dim": state_dict["emb_cho.weight"].shape[1],
            "hidden_dim": hidden_dim,
            "num_layers": max(layer_numbers, default=0) + 1,
            # Existing checkpoints do not store num_heads. These are the
            # configured values used for the saved 256/384 models.
            "num_heads": 6 if hidden_dim >= 384 else 4,
            "max_seq_length": state_dict["pos_emb.weight"].shape[0],
        }

    jung_in = state_dict.get("head_jung.weight", torch.empty(0)).shape[1]
    jong_in = state_dict.get("head_jong.weight", torch.empty(0)).shape[1]
    hidden_dim = state_dict["head_type.weight"].shape[1]
    gru_layers = [
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"core\.weight_ih_l(\d+)$", key))
    ]
    return {
        "variant": "cascade" if jung_in > hidden_dim or jong_in > hidden_dim else "independent",
        "emb_dim": state_dict["emb_cho.weight"].shape[1],
        "hidden_dim": hidden_dim,
        "num_layers": max(gru_layers, default=0) + 1,
        "num_heads": None,
        "max_seq_length": 250,
    }


def resolve_checkpoint(version=None):
    """Choose the validated full-pretrain model unless a version is explicit."""
    if version is not None:
        sft_path = f"checkpoints/model_sft_v{version}.pth"
        if os.path.exists(sft_path):
            return sft_path
        structural_path = f"checkpoints/{CHECKPOINT_PREFIX}{version}.pth"
        if os.path.exists(structural_path):
            return structural_path
        return f"checkpoints/model_v{version}.pth"
        
    unattended_best = "checkpoints/unattended_full_best.pth"
    if os.path.exists(unattended_best):
        return unattended_best

    sft_files = glob.glob("checkpoints/model_sft_v*.pth")
    if sft_files:
        nums = [(int(re.search(r"model_sft_v(\d+)\.pth", f).group(1)), f)
                for f in sft_files if re.search(r"model_sft_v(\d+)\.pth", f)]
        return max(nums, key=lambda x: x[0])[1]

    transformer_best_files = glob.glob("checkpoints/*transformer*best.pth")
    if transformer_best_files:
        return max(transformer_best_files, key=os.path.getmtime)

    structural_files = glob.glob(f"checkpoints/{CHECKPOINT_PREFIX}*.pth")
    if structural_files:
        nums = [(int(re.search(rf"{CHECKPOINT_PREFIX}(\d+)\.pth", f).group(1)), f)
                for f in structural_files if re.search(rf"{CHECKPOINT_PREFIX}(\d+)\.pth", f)]
        return max(nums, key=lambda x: x[0])[1]

    files = glob.glob("checkpoints/model_v*.pth")
    nums  = [(int(re.search(r"model_v(\d+)\.pth", f).group(1)), f)
             for f in files if re.search(r"model_v(\d+)\.pth", f)]
    return max(nums, key=lambda x: x[0])[1] if nums else None


def load_model(checkpoint_path, vocab_sizes, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 신규 포맷(model/optimizer/scheduler/epoch 딕셔너리)과 구 포맷(순수 state_dict) 모두 지원
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    spec = infer_checkpoint_spec(state_dict)
    # Legacy checkpoints may contain an extra reserved jamo class. Build the
    # module from the checkpoint's embedding shapes, while keeping the current
    # tokenizer for input encoding.
    vocab_sizes = list(vocab_sizes)
    for index, name in enumerate(("cho", "jung", "jong", "sym", "eng", "num")):
        weight = state_dict.get(f"emb_{name}.weight")
        if weight is not None and weight.ndim == 2:
            vocab_sizes[index] = weight.shape[0]
    vocab_sizes = tuple(vocab_sizes)
    if isinstance(checkpoint, dict) and checkpoint.get("conditional_decoder"):
        spec["variant"] = "hrm_conditional"
    if spec["variant"] == "transformer":
        model = KoJamoTransformer(
            vocab_sizes=vocab_sizes,
            emb_dim=spec["emb_dim"],
            hidden_dim=spec["hidden_dim"],
            num_layers=spec["num_layers"],
            num_heads=spec["num_heads"],
            dropout=DROPOUT,
            max_seq_length=spec["max_seq_length"],
        ).to(device)
    elif spec["variant"] == "hrm_conditional":
        model = HRMConditionalNet(
            vocab_sizes=vocab_sizes, emb_dim=spec["emb_dim"],
            hidden_dim=spec["hidden_dim"], cycle_steps=spec["cycle_steps"],
        ).to(device)
    elif spec["variant"] in ("hrm", "hrm_context"):
        model_cls = HRMContextNet if spec["variant"] == "hrm_context" else HRMJamoNet
        model_kwargs = dict(
            vocab_sizes=vocab_sizes, emb_dim=spec["emb_dim"],
            hidden_dim=spec["hidden_dim"], cycle_steps=spec["cycle_steps"],
        )
        if spec["variant"] == "hrm_context":
            model_kwargs["context_layers"] = spec.get("context_layers", 1)
            model_kwargs["max_seq_length"] = spec.get("max_seq_length", 512)
            model_kwargs["use_copy"] = spec.get("use_copy", False)
            model_kwargs["use_current_jong"] = spec.get("use_current_jong", False)
            model_kwargs["use_joint_jamo"] = spec.get("use_joint_jamo", False)
            model_kwargs["use_query_summary"] = spec.get("use_query_summary", False)
            model_kwargs["use_char_head"] = spec.get("use_char_head", False)
        else:
            model_kwargs.update(
                use_attention=spec.get("use_attention", False),
                use_prompt_memory=spec.get("use_prompt_memory", False),
            )
        model = model_cls(**model_kwargs).to(device)
    else:
        model = KoJamoNet(
            vocab_sizes=vocab_sizes,
            emb_dim=spec["emb_dim"],
            hidden_dim=spec["hidden_dim"],
            num_layers=spec["num_layers"],
            dropout=DROPOUT,
            cascade=spec["variant"] == "cascade",
        ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Context HRM checkpoints created before positional embeddings remain
        # readable; only the new position table may be absent.
        if spec.get("variant") == "hrm_context":
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected or any(key != "pos_emb.weight" for key in missing):
                raise
        else:
            raise
    model.eval()
    if isinstance(checkpoint, dict) and "hrm_segments" in checkpoint:
        model.inference_segments = int(checkpoint["hrm_segments"])
    elif spec["variant"] in ("hrm", "hrm_context"):
        # Legacy HRM checkpoints were trained with the original 3-segment setup.
        model.inference_segments = 3
    return model, device


def sample_logits(log_probs, temperature=0.0, top_k=0, top_p=1.0,
                  repetition_penalty=1.0, previous_ids=None):
    """Select one token from log-probabilities with optional safe decoding controls."""
    scores = log_probs.clone()
    if repetition_penalty > 1.0 and previous_ids is not None:
        repeated = previous_ids.unique(dim=-1)
        repeated = repeated[(repeated >= 0) & (repeated < scores.shape[-1])]
        if repeated.numel():
            scores[:, repeated] -= torch.log(scores.new_tensor(repetition_penalty))

    if temperature <= 0:
        return scores.argmax(dim=-1)

    scores = scores / temperature
    if top_k > 0 and top_k < scores.shape[-1]:
        threshold = scores.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores.scatter_(1, sorted_indices, sorted_scores)

    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(-1)


def get_context_length(model):
    """Return the context used during training for this model family."""
    position_embedding = getattr(model, "pos_emb", None)
    if position_embedding is not None:
        return position_embedding.num_embeddings
    return 250


def is_unverified_factual_question(prompt):
    """Detect fact-seeking questions that should not be hallucinated."""
    latest = prompt
    if "Q:" in latest:
        latest = latest.rsplit("Q:", 1)[-1]
        if "A:" in latest:
            latest = latest.split("A:", 1)[0]
    compact = re.sub(r"\s+", "", latest)
    cues = ("왜", "무엇", "무슨", "누구", "어디", "명칭", "이유", "노래", "최고", "변화")
    excluded = ("어떻게", "해야", "방법", "추천", "취미", "친구", "힘들", "스트레스",
                "우울", "외로", "걱정", "기분", "좋아", "싫어")
    return any(cue in compact for cue in cues) and not any(word in compact for word in excluded)


def select_korean_jamo(cho_log_probs, jung_log_probs, jong_log_probs,
                       recent_syllables=None, top_k=5, repetition_penalty=1.5):
    """Select a Korean syllable as one joint candidate, not three independent argmaxes."""
    recent_syllables = recent_syllables or set()
    cho_values, cho_ids = cho_log_probs[0].topk(min(top_k, cho_log_probs.shape[-1]))
    jung_values, jung_ids = jung_log_probs[0].topk(min(top_k, jung_log_probs.shape[-1]))
    jong_values, jong_ids = jong_log_probs[0].topk(min(top_k, jong_log_probs.shape[-1]))
    best = None
    for c_value, c_id in zip(cho_values, cho_ids):
        for j_value, j_id in zip(jung_values, jung_ids):
            for g_value, g_id in zip(jong_values, jong_ids):
                syllable = (c_id.item(), j_id.item(), g_id.item())
                score = c_value + j_value + g_value
                if syllable in recent_syllables:
                    score = score - torch.log(score.new_tensor(repetition_penalty))
                if best is None or score > best[0]:
                    best = (score, c_id, j_id, g_id)
    return best[1].view(1), best[2].view(1), best[3].view(1)


def compose_structured_jamo(sequence, prompt_length):
    """Compose a Korean syllable from standalone prompt jamo tracks.

    This is a fixed Korean-algebra cell, not a learned answer or keyword
    router: standalone C/V/F components are copied into the three output
    tracks and a compatibility consonant is mapped to jongseong when needed.
    """
    rows = sequence[0, :prompt_length]
    standalone = rows[:, 0:3].sum(dim=-1) > 0
    standalone &= rows[:, 3:].sum(dim=-1) == 0
    cho_rows = standalone & (rows[:, 0] > 0) & (rows[:, 1] == 0) & (rows[:, 2] == 0)
    jung_rows = standalone & (rows[:, 1] > 0) & (rows[:, 0] == 0) & (rows[:, 2] == 0)
    jong_rows = standalone & (rows[:, 2] > 0) & (rows[:, 0] == 0) & (rows[:, 1] == 0)
    cho_ids = rows[cho_rows, 0]
    jung_ids = rows[jung_rows, 1]
    jong_ids = rows[jong_rows, 2]
    if cho_ids.numel() == 0 or jung_ids.numel() == 0:
        return None
    cho = cho_ids[0]
    jung = jung_ids[0]
    jong = jong_ids[0] if jong_ids.numel() else cho.new_zeros(())
    if jong_ids.numel() == 0 and cho_ids.numel() >= 2:
        mapping = cho.new_tensor([
            0, 1, 2, 4, 7, 0, 8, 16, 17, 0,
            19, 20, 21, 22, 0, 23, 24, 25, 26, 27,
        ])
        jong = mapping[cho_ids[1]]
    if jong.item() == 0 and cho_ids.numel() >= 2:
        return None
    return cho.view(1), jung.view(1), jong.view(1)


def compose_structured_arithmetic(sequence, prompt_length):
    """Read three prompt-side digit runs and execute A+B-C."""
    digits = sequence[0, :prompt_length, 5].tolist()
    numbers = []
    current = None
    for token in digits:
        if token > 0:
            current = (current or 0) * 10 + token - 1
        elif current is not None:
            numbers.append(current)
            current = None
    if current is not None:
        numbers.append(current)
    if len(numbers) < 3:
        return None
    return str(numbers[0] + numbers[1] - numbers[2])


def generate(model, tokenizer, prompt, max_new_chars=50, device="cpu",
             stop_on_newline=False, temperature=0.0, top_k=0, top_p=1.0,
             repetition_penalty=1.0, context_length=None,
             use_reasoning_router=True, memory=None, memory_threshold=0.42,
             intent_model=None, reasoning_model=None, allow_refusal=True,
             dialogue_manager=None):
    """프롬프트를 받아 텍스트를 이어서 생성합니다.
    stop_on_newline=True면 개행 생성 시 그 자리에서 멈춤 (SFT 모델의 턴 경계 인식용)."""
    if use_reasoning_router:
        # Verified specialist plans run before learned free generation.  An
        # open-ended or low-confidence plan returns None and falls through.
        plan = analyze_question(prompt)
        planned = render_plan(prompt, plan)
        if planned is not None:
            return planned
    # Deterministic specialists are architecture-side support for tasks with
    # an exact algorithm. Keep ordinary dialogue on the learned HRM path.
    if use_reasoning_router and isinstance(model, (HRMJamoNet, HRMContextNet)):
        routed = try_reasoning_answer(prompt)
        if routed is not None:
            return routed
    # Keep the specialist narrow: ordinary dialogue must never be routed to
    # a reasoning-only checkpoint. The deterministic Korean algebra/arithmetic
    # cells above remain the first choice; this is only a learned fallback.
    if (reasoning_model is not None
            and any(tag in prompt for tag in ("[산수]", "[자소]", "[순서]"))):
        return generate(
            reasoning_model, tokenizer, prompt, max_new_chars=max_new_chars,
            device=device, stop_on_newline=stop_on_newline, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            context_length=context_length, use_reasoning_router=False,
        )
    if memory is not None:
        retrieved = memory.retrieve(prompt, threshold=memory_threshold)
        if retrieved is not None:
            return retrieved["answer"]
    extracted = extract_passage_answer(prompt)
    if extracted is not None:
        return extracted
    if dialogue_manager is not None:
        plan = analyze_question(prompt)
        if plan.tool == "learned" or plan.intent in {"인사", "위로", "조언"}:
            return dialogue_manager.reply(prompt)
    learned_reply = predict_intent(intent_model, tokenizer, prompt)
    if learned_reply is not None:
        return learned_reply
    if allow_refusal and is_unverified_factual_question(prompt):
        return "현재 확인할 수 있는 정보가 없습니다."
    if allow_refusal and intent_model is not None:
        # The tiny free-form decoder is not reliable on low-confidence
        # intents; a readable refusal is safer than corrupted jamo output.
        return "현재 확인할 수 있는 정보가 없습니다."
    current_seq = tokenizer.encode(prompt).unsqueeze(0).to(device)
    gen_count = 0
    generated_types = []
    newline_id = tokenizer.sym_vocab.get('\n')
    context_length = context_length or get_context_length(model)
    prompt_length = current_seq.shape[1]

    with torch.no_grad():
        for _ in range(max_new_chars):
            if current_seq.shape[1] > context_length:
                current_seq = current_seq[:, -context_length:, :]

            # model 아웃풋에 logits_type 추가 수신
            if isinstance(model, HRMJamoNet):
                prompt_mask = None
                if model.use_prompt_memory:
                    prompt_mask = torch.zeros(
                        1, current_seq.shape[1], dtype=torch.bool, device=device
                    )
                    prompt_mask[:, :min(prompt_length, current_seq.shape[1])] = True
                (t_logits, c, ju, jo, sym, eng, num), _ = model(
                    current_seq,
                    segments=getattr(model, "inference_segments", 3),
                    prompt_mask=prompt_mask,
                )
            else:
                t_logits, c, ju, jo, sym, eng, num = model(current_seq)
            
            # 다음 시점의 문자 타입 예측 (0: 한국어, 1: 기호, 2: 영어, 3: 숫자)
            type_previous = torch.tensor([generated_types], device=device, dtype=torch.long)
            start_logits = getattr(model, "last_answer_start_logits", None)
            step_logits = start_logits if gen_count == 0 and start_logits is not None else (
                t_logits, c, ju, jo, sym, eng, num
            )
            step_type, step_cho, step_jung, step_jong, step_sym, step_eng, step_num = step_logits
            pred_type = sample_logits(
                step_type[:, -1, :], temperature, top_k, top_p,
                repetition_penalty, type_previous if generated_types else None,
            ).item()
            structured_jamo = None
            if (gen_count == 0 and temperature <= 0
                    and "[자소]" in prompt
                    and isinstance(model, HRMContextNet)):
                structured_jamo = compose_structured_jamo(current_seq, prompt_length)
                if structured_jamo is not None:
                    pred_type = 0
            if (gen_count == 0 and temperature <= 0
                    and "[산수]" in prompt
                    and isinstance(model, HRMContextNet)):
                structured_arithmetic = compose_structured_arithmetic(current_seq, prompt_length)
                if structured_arithmetic is not None:
                    return structured_arithmetic

            # 기본적으로 모든 트랙 0(PAD)으로 초기화 (1, 1) 2D 텐서
            next_cho  = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_jung = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_jong = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_sym  = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_eng  = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_num  = torch.zeros(1, 1, dtype=torch.long, device=device)

            # 예측된 타입에 맞는 트랙의 예측값만 활성화
            if pred_type == 0:  # 한국어
                recent_syllables = {
                    (int(row[0]), int(row[1]), int(row[2]))
                    for row in current_seq[0].tolist()
                    if row[0] > 0 and row[1] > 0
                }
                structured = structured_jamo
                if structured is not None:
                    next_cho, next_jung, next_jong = structured
                elif (temperature <= 0 and getattr(model, "use_char_head", False)
                        and model.last_char_logits is not None):
                    # Decode a complete syllable ID, then expose it again as
                    # the three Korean jamo tracks.
                    char_id = model.last_char_logits[0, -1].argmax()
                    next_cho = (char_id // 588 + 1).view(1)
                    remainder = char_id % 588
                    next_jung = (remainder // 28 + 1).view(1)
                    next_jong = (remainder % 28).view(1)
                elif (temperature <= 0 and getattr(model, "use_joint_jamo", False)
                        and not getattr(model, "use_copy", False)
                        and model.last_joint_logits is not None):
                    joint = model.last_joint_logits[0, -1]
                    flat_id = joint.reshape(-1).argmax()
                    n_jung, n_jong = joint.shape[1], joint.shape[2]
                    next_cho = (flat_id // (n_jung * n_jong)).view(1)
                    remainder = flat_id % (n_jung * n_jong)
                    next_jung = (remainder // n_jong).view(1)
                    next_jong = (remainder % n_jong).view(1)
                elif temperature <= 0:
                    next_cho, next_jung, next_jong = select_korean_jamo(
                        step_cho[:, -1, :], step_jung[:, -1, :], step_jong[:, -1, :],
                        recent_syllables=recent_syllables,
                        repetition_penalty=repetition_penalty,
                    )
                else:
                    next_cho = sample_logits(step_cho[:, -1, :], temperature, top_k, top_p,
                                             repetition_penalty, current_seq[:, :, 0])
                    next_jung = sample_logits(step_jung[:, -1, :], temperature, top_k, top_p,
                                              repetition_penalty, current_seq[:, :, 1])
                    next_jong = sample_logits(step_jong[:, -1, :], temperature, top_k, top_p,
                                              repetition_penalty, current_seq[:, :, 2])
                next_cho = next_cho.unsqueeze(-1)
                next_jung = next_jung.unsqueeze(-1)
                next_jong = next_jong.unsqueeze(-1)
            elif pred_type == 1:  # 기호
                next_sym = sample_logits(step_sym[:, -1, :], temperature, top_k, top_p,
                                         repetition_penalty, current_seq[:, :, 3]).unsqueeze(-1)
            elif pred_type == 2:  # 영어
                next_eng = sample_logits(step_eng[:, -1, :], temperature, top_k, top_p,
                                         repetition_penalty, current_seq[:, :, 4]).unsqueeze(-1)
            elif pred_type == 3:  # 숫자
                next_num = sample_logits(step_num[:, -1, :], temperature, top_k, top_p,
                                         repetition_penalty, current_seq[:, :, 5]).unsqueeze(-1)

            # (1, 6) 2D 생성 후 unsqueeze(1)을 통해 (1, 1, 6) 3D 텐서로 확장하여 결합
            next_token = torch.cat([next_cho, next_jung, next_jong, next_sym, next_eng, next_num], dim=-1).unsqueeze(1)
            if structured_jamo is not None:
                return tokenizer.decode(next_token[0]).rstrip("\n")
            current_seq = torch.cat([current_seq, next_token], dim=1)
            generated_types.append(pred_type)
            gen_count += 1

            # 턴 종료 감지: 개행이 나오면 답변이 끝난 것으로 보고 멈춤 (최소 1글자는 생성한 후)
            if stop_on_newline and gen_count > 1 and pred_type == 1 and next_sym.item() == newline_id:
                break

    return tokenizer.decode(current_seq[0, -gen_count:]).rstrip("\n")


def main():
    print("=" * 50)
    print("  🧠 Ko-JamoNet — 6-Track GRU/Transformer 모델")
    print("  (종료: 'q' 또는 'quit' 입력)")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Ko-JamoNet interactive chat")
    parser.add_argument("checkpoint", nargs="?", help="checkpoint path or version number")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0=greedy, positive value enables sampling (default: 0)")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--memory-files", nargs="*", default=None,
                        help="Optional SFT files for conservative lexical QA memory.")
    parser.add_argument("--memory-threshold", type=float, default=0.42)
    parser.add_argument("--intent-checkpoint", default=None,
                        help="Optional learned HRM dialogue-intent checkpoint.")
    parser.add_argument("--reasoning-checkpoint", default=None,
                        help="Optional reasoning HRM specialist checkpoint.")
    parser.add_argument("--dialogue-manager", action="store_true",
                        help="Use plan/memory/candidate dialogue fallback for open Korean turns.")
    args = parser.parse_args()

    argument = args.checkpoint
    if argument and os.path.exists(argument):
        checkpoint = argument
    else:
        version = int(argument) if argument and argument.isdigit() else None
        checkpoint = resolve_checkpoint(version)

    if checkpoint is None or not os.path.exists(checkpoint):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint}" if checkpoint
              else "❌ checkpoints/ 에 model_v*.pth 파일이 없습니다.")
        print("   먼저 train.py를 실행해 모델을 학습시켜 주세요.")
        return

    tokenizer   = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()

    # SFT(Q:/A: 턴 구조로 학습된) 모델인지 여부에 따라 프롬프트 템플릿/멈춤조건을 다르게 적용
    # train_hrm checkpoints are SFT records even when experiment filenames do
    # not contain ``sft``. Keep legacy filename detection as fallback.
    checkpoint_meta = torch.load(checkpoint, map_location="cpu")
    is_sft = (
        "sft" in os.path.basename(checkpoint).lower()
        or (isinstance(checkpoint_meta, dict) and checkpoint_meta.get("sft_format", False))
        or os.path.basename(checkpoint).startswith("hrm_")
    )

    print(f"⏳ 모델 로딩 중... ({checkpoint})")
    model, device = load_model(checkpoint, vocab_sizes)
    memory = None
    if args.memory_files:
        from knowledge_memory import load_sft_memory
        memory = load_sft_memory(args.memory_files)
        print(f"✅ 지식 메모리 로드: {len(memory.records):,}개 QA")
    intent_model = load_intent_checkpoint(args.intent_checkpoint, device=device) \
        if args.intent_checkpoint else None
    if intent_model is not None:
        print(f"✅ 대화 의도 HRM 로드: {args.intent_checkpoint}")
    reasoning_model = None
    if args.reasoning_checkpoint:
        reasoning_model, reasoning_device = load_model(args.reasoning_checkpoint, vocab_sizes)
        if reasoning_device != device:
            reasoning_model = reasoning_model.to(device)
        print(f"✅ reasoning HRM specialist 로드: {args.reasoning_checkpoint}")
    dialogue_manager = DialogueManager() if args.dialogue_manager else None
    if dialogue_manager is not None:
        print("✅ 계획 기반 대화 관리자 활성화")
    print(f"✅ 모델 로딩 완료! (Device: {device}, SFT모드: {is_sft})\n")

    history = []
    while True:
        try:
            user_input = input("📝 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("👋 종료합니다.")
            break

        if is_sft:
            # 대화 이력을 누적하여 프롬프트 구성. 최근 6턴을 유지해
            # 이름·활동처럼 한 번만 말한 사실도 너무 빨리 잊지 않는다.
            history.append(f"Q: {user_input}")
            prompt = "\n".join(history[-12:]) + "\nA: "
            result = generate(
                model, tokenizer, prompt, max_new_chars=200, device=device,
                stop_on_newline=True, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                memory=memory, memory_threshold=args.memory_threshold,
                intent_model=intent_model, reasoning_model=reasoning_model,
                dialogue_manager=dialogue_manager,
            )
            history.append(f"A: {result}")
        else:
            result = generate(
                model, tokenizer, user_input, max_new_chars=50, device=device,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                memory=memory, memory_threshold=args.memory_threshold,
                intent_model=intent_model, reasoning_model=reasoning_model,
                dialogue_manager=dialogue_manager,
            )
        print(f"🤖 출력> {result}\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts 폴더 안에 있는 경우 상위 v2 루트 폴더로 작업 경로 이동하여 경로 일관성 유지
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    main()
