import torch
import pytest
from model import KoJamoNet
from tokenizer import KoJamoTokenizer

def test_pointer_generator():
    print("🧪 Pointer-Generator 구조 진단 및 통합 테스트 시작...")
    
    tokenizer = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 테스트 디바이스: {device}")
    
    # 1. 모델 인스턴스화
    print("1. 모델 인스턴스화 테스트 중...")
    model = KoJamoNet(vocab_sizes=vocab_sizes, emb_dim=64, hidden_dim=256, num_layers=2).to(device)
    if not hasattr(model, "head_copy"):
        pytest.skip("legacy pointer test requires KoJamoNet head_copy, but current model has no pointer head")
    print("✅ 모델 인스턴스화 완료.")
    
    # 2. 더미 입력 데이터 준비 (Batch=2, Seq=10)
    print("2. 더미 데이터 입력 및 순방향 연산 테스트 중...")
    dummy_text_1 = "가나다라"
    dummy_text_2 = "abcdefgh"
    
    x1 = tokenizer.encode(dummy_text_1)
    x2 = tokenizer.encode(dummy_text_2)
    
    # 길이를 맞추기 위해 패딩 처리
    max_len = max(len(x1), len(x2))
    padded_x = []
    for x in [x1, x2]:
        if len(x) < max_len:
            padding = torch.zeros(max_len - len(x), 6, dtype=torch.long)
            padded_x.append(torch.cat([x, padding], dim=0))
        else:
            padded_x.append(x)
            
    x_input = torch.stack(padded_x, dim=0).to(device) # [2, max_len, 6]
    print(f"👉 입력 텐서 형태: {x_input.shape}")
    
    # 3. 학습 모드 (Teacher Forcing 존재) 테스트
    print("3. 교사 강요(Teacher Forcing) 순방향 연산 수행 중...")
    log_final_type, log_final_cho, log_final_jung, log_final_jong, log_final_sym, log_final_eng, log_final_num = model(x_input, target_for_forcing=x_input)
    
    print(f"✅ 교사 강요 연산 완료.")
    print(f"   - log_final_cho 형태: {log_final_cho.shape}")
    
    # 확률 유효성 검증 (sum = 1)
    for name, log_prob in [("Type", log_final_type), ("Cho", log_final_cho), ("Jung", log_final_jung), 
                           ("Jong", log_final_jong), ("Sym", log_final_sym), ("Eng", log_final_eng), ("Num", log_final_num)]:
        prob = torch.exp(log_prob)
        prob_sum = prob.sum(dim=-1)
        # sum이 거의 1인지 확인 (수치적 허용오차 1e-4)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4), f"❌ {name} 확률의 합이 1이 아닙니다: {prob_sum}"
    print("✅ 모든 출력 트랙의 확률 분포 유효성 검증 통과 (합 = 1).")
    
    # 4. 추론 모드 (Teacher Forcing 없음) 테스트
    print("4. 추론 모드(교사 강요 없음) 순방향 연산 수행 중...")
    log_final_type, log_final_cho, log_final_jung, log_final_jong, log_final_sym, log_final_eng, log_final_num = model(x_input)
    print("✅ 추론 모드 연산 완료.")
    
    # 5. 오차 역전파 및 경사도 계산 검증
    print("5. 오차 역전파(Backward) 및 가중치 업데이트 검증 중...")
    loss = torch.exp(log_final_cho).sum() # 더미 Loss
    loss.backward()
    
    # head_copy 경사도 확인
    has_grad = model.head_copy.weight.grad is not None
    assert has_grad, "❌ head_copy 가중치의 경사도가 계산되지 않았습니다."
    print("✅ head_copy 레이어 역전파 경사도 검증 완료.")
    print("🎉 모든 진단 테스트 성공! 구조적 업그레이드가 완벽히 작동함.")

if __name__ == "__main__":
    test_pointer_generator()
