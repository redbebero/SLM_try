import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from tokenizer import KoJamoTokenizer
from chat import load_model

def debug_generation():
    checkpoint = "checkpoints/model_sft_v124.pth"
    tokenizer = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()
    
    model, device = load_model(checkpoint, vocab_sizes)
    model.eval()
    
    # 평가 샘플 1
    context1 = "리슐리외는 프랑스 루이 13세 시기의 총리이다. 그는 중앙집권체제의 확립과 왕권 강화에 힘을 쏟았으며 절대왕정의 기초를 쌓았다."
    question1 = "절대왕정의 기초를 쌓은 인물은 누구인가?"
    prompt1 = f"Q: 지문: {context1} 질문: {question1}\nA: "
    
    input_ids = tokenizer.encode(prompt1).unsqueeze(0).to(device)
    print("입력 ID 크기:", input_ids.shape)
    
    # 30 스텝 포워드하면서 p_copy 값 추적
    current_seq = input_ids.clone()
    
    print("\n--- 생성 단계별 p_copy 및 어텐션 상태 디버깅 ---")
    for step in range(40):
        with torch.no_grad():
            # forward에서 중간 값들을 모니터링하기 위해 직접 실행
            # 1. 임베딩 및 RNN 통과
            types = model._get_types(current_seq)
            type_emb = model.emb_type(types)
            prev_jong = torch.cat([
                torch.zeros(current_seq.size(0), 1, dtype=torch.long, device=current_seq.device),
                current_seq[:, :-1, 2]
            ], dim=1)
            prev_jong_emb = model.emb_jong(prev_jong)
            
            merged = torch.cat([
                model.emb_cho(current_seq[:, :, 0]),
                model.emb_jung(current_seq[:, :, 1]),
                model.emb_jong(current_seq[:, :, 2]),
                model.emb_sym(current_seq[:, :, 3]),
                model.emb_eng(current_seq[:, :, 4]),
                model.emb_num(current_seq[:, :, 5]),
                type_emb,
                prev_jong_emb
            ], dim=-1)
            
            core_input = model.proj_in(merged)
            core_state, _ = model.core(core_input)
            
            # 2. 어텐션
            seq_len = core_state.size(1)
            pos = model._pos_cache[:seq_len]
            causal_mask = model._causal_mask_cache[:seq_len, :seq_len]
            attn_input = model.pre_attn_norm(core_state + pos.unsqueeze(0))
            attn_out, attn_weights = model.attn(attn_input, attn_input, attn_input, attn_mask=causal_mask, need_weights=True)
            
            # 3. p_copy
            p_copy_all = torch.sigmoid(model.head_copy(core_state)) # [B, S, 1]
            p_copy = p_copy_all[0, -1, 0].item() # 마지막 스텝의 복사 확률
            
            # 4. 다음 문자 생성 로직
            # 각 헤드 통과
            logits_type = model.head_type(core_state)
            logits_cho = model.head_cho(core_state)
            logits_jung = model.head_jung(torch.cat([core_state, model.emb_cho(current_seq[:, :, 0])], dim=-1))
            logits_jong = model.head_jong(torch.cat([core_state, model.emb_cho(current_seq[:, :, 0]), model.emb_jung(current_seq[:, :, 1])], dim=-1))
            logits_sym = model.head_sym(core_state)
            logits_eng = model.head_eng(core_state)
            logits_num = model.head_num(core_state)
            
            # 확률 계산
            p_vocab_cho = torch.softmax(logits_cho, dim=-1)
            p_copy_cho = model._get_copy_prob(attn_weights, current_seq[:, :, 0], logits_cho.size(-1))
            p_final_cho = (1.0 - p_copy_all) * p_vocab_cho + p_copy_all * p_copy_cho
            
            # 디코딩 결과 확인
            pred_cho = p_final_cho[0, -1].argmax().item()
            pred_cho_char = tokenizer.cho_list[pred_cho] if pred_cho < len(tokenizer.cho_list) else "?"
            
            # 어텐션 최대 활성 위치
            max_attn_idx = attn_weights[0, -1].argmax().item()
            # 해당 위치의 입력 자소 디코딩
            attended_input = tokenizer.decode(current_seq[0, max_attn_idx:max_attn_idx+1])
            
            print(f"Step {step:02d} | p_copy: {p_copy:.4f} | Max Attn Pos: {max_attn_idx} ('{attended_input}') | Pred Cho: '{pred_cho_char}'")
            
            # 다음 시퀀스 갱신 (chat.py처럼 argmax 기반 추가)
            next_char = torch.zeros(1, 1, 6, dtype=torch.long, device=device)
            next_char[0, 0, 0] = p_final_cho[0, -1].argmax().item()
            # 다른 트랙들도 유사하게 계산하여 시퀀스 갱신
            next_char[0, 0, 1] = ((1.0 - p_copy_all) * torch.softmax(logits_jung, dim=-1) + p_copy_all * model._get_copy_prob(attn_weights, current_seq[:, :, 1], logits_jung.size(-1)))[0, -1].argmax().item()
            next_char[0, 0, 2] = ((1.0 - p_copy_all) * torch.softmax(logits_jong, dim=-1) + p_copy_all * model._get_copy_prob(attn_weights, current_seq[:, :, 2], logits_jong.size(-1)))[0, -1].argmax().item()
            next_char[0, 0, 3] = ((1.0 - p_copy_all) * torch.softmax(logits_sym, dim=-1) + p_copy_all * model._get_copy_prob(attn_weights, current_seq[:, :, 3], logits_sym.size(-1)))[0, -1].argmax().item()
            next_char[0, 0, 4] = ((1.0 - p_copy_all) * torch.softmax(logits_eng, dim=-1) + p_copy_all * model._get_copy_prob(attn_weights, current_seq[:, :, 4], logits_eng.size(-1)))[0, -1].argmax().item()
            next_char[0, 0, 5] = ((1.0 - p_copy_all) * torch.softmax(logits_num, dim=-1) + p_copy_all * model._get_copy_prob(attn_weights, current_seq[:, :, 5], logits_num.size(-1)))[0, -1].argmax().item()
            
            current_seq = torch.cat([current_seq, next_char], dim=1)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    debug_generation()
