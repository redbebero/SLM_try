import torch
from tokenizers import Tokenizer
from model import MicroHRMDeepSeek
import os

def generate_answer(model, tokenizer, prompt, max_new_tokens=500, device="cpu"):
    model.eval()
    
    # DPO 학습 데이터셋 구조와 완벽히 동일하게 맞춤
    prompt_text = f"[문제]\n{prompt}\n\n[정답 풀이 (Chosen)]\n"
    input_ids = tokenizer.encode(prompt_text).ids
    
    # 텐서로 변환
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    eos_id = tokenizer.token_to_id("<eos>")
    
    print("\n--- [추론 결과] ---")
    print(prompt_text, end="")
    
    generated_ids = []
    current_text = ""
    
    with torch.no_grad():
        past_kv = None
        current_input = input_tensor
        
        for _ in range(max_new_tokens):
            if current_input.size(1) >= model.max_seq_len:
                print("\n[경고: 최대 시퀀스 길이에 도달하여 생성을 중단합니다.]", end="")
                break
                
            # 순전파 (KV Cache 버그 회피를 위해 전체 시퀀스를 다시 입력)
            _, logits = model(current_input, state=None, use_kv_cache=False)
            
            # 마지막 토큰의 로짓 가져오기
            next_token_logits = logits[0, -1, :]
            
            # Greedy Decoding (가장 확률이 높은 토큰만 선택)
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token_id == eos_id:
                break
                
            generated_ids.append(next_token_id)
            
            # 디코딩 후 실시간 출력
            new_text = tokenizer.decode(generated_ids)
            if len(new_text) > len(current_text):
                print(new_text[len(current_text):], end="", flush=True)
                current_text = new_text
            
            # 다음 입력을 위해 생성된 토큰 추가 (전체 시퀀스 입력)
            current_input = torch.cat([current_input, torch.tensor([[next_token_id]], device=device)], dim=1)
            
    print("\n-------------------\n")
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading on {device}...")
    
    # 1. 토크나이저 로드
    tokenizer_path = "custom_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} 가 없습니다. 먼저 훈련(train.py)을 진행해주세요.")
        exit(1)
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # BPE ByteLevel 디코더 설정 (외계어가 정상적인 한국어로 복원되도록 함)
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()
    # 2. 모델 구조 초기화 (학습할 때와 정확히 동일한 설정)
    model = MicroHRMDeepSeek(
        vocab_size=16000, 
        dim=768, 
        num_heads=12, 
        latent_dim=64, 
        mlp_hidden_dim=3072, 
        max_seq_len=512, 
        thinking_steps=5 
    ).to(device)
    
    # 3. 학습된 가중치(Weights) 불러오기 (원하는 에폭 또는 최신 에폭 자동 로드)
    import glob
    checkpoints = glob.glob("checkpoints/micro_hrm_deepseek_ep*.pt")
    if checkpoints:
        latest_weight = max(checkpoints, key=lambda x: int(x.split('_ep')[1].split('.pt')[0]))
        
        epoch_input = input("\n불러올 에폭(숫자)을 입력하세요 (엔터 시 최신 에폭 로드): ").strip()
        
        target_weight = latest_weight
        if epoch_input.isdigit():
            requested_weight = f"checkpoints/micro_hrm_deepseek_ep{epoch_input}.pt"
            if os.path.exists(requested_weight):
                target_weight = requested_weight
            else:
                print(f"⚠️ '{requested_weight}' 파일이 없습니다. 최신 에폭으로 대체합니다.")
        elif epoch_input != "":
            print(f"⚠️ 유효하지 않은 입력입니다. 최신 에폭으로 대체합니다.")
            
        state_dict = torch.load(target_weight, map_location=device, weights_only=True)
        # torch.compile로 인해 추가된 '_orig_mod.' 접두사 제거
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
        print(f"✅ {target_weight} 모델 가중치를 성공적으로 불러왔습니다.")
    else:
        print(f"⚠️ 저장된 가중치 파일이 없습니다. 초기화된 무작위 가중치로 테스트합니다.")

    # 4. 사용자와의 대화 루프 (추론 테스트)
    print("\n💡 테스트할 문제를 자연스럽게 입력하세요. 종료하려면 'exit'를 입력하세요.")
    while True:
        print("\n" + "="*50)
        prompt = input("문제를 입력하세요: ").replace('\\n', '\n')
        if prompt.strip().lower() == 'exit':
            break
            
        generate_answer(model, tokenizer, prompt=prompt, device=device)

