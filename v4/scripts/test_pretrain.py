import torch
import sys
import os
import pytest

# scripts 폴더 임포트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from tokenizer import KoJamoTokenizer
from chat import load_model, generate

def test():
    # v2에서 복사해온 사전학습 가중치 지정
    checkpoint = "checkpoints/model_v13.pth"
    if not os.path.exists(checkpoint):
        pytest.skip(f"optional legacy checkpoint missing: {checkpoint}")
    tokenizer = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()
    
    print(f"⏳ 사전학습 모델 로딩 중: {checkpoint}")
    model, device = load_model(checkpoint, vocab_sizes)
    
    # 일반 사전학습 프롬프트로 한글 조합력 테스트
    prompts = ["대한민국은 ", "이순신 장군은 ", "세종대왕이 "]
    
    print("\n=== 사전학습 모델 한글 생성력 테스트 ===")
    for p in prompts:
        # SFT가 아니므로 stop_on_newline=False, 일반 생성
        result = generate(model, tokenizer, p, max_new_chars=30, device=device, stop_on_newline=False)
        print(f"입력: {p}")
        print(f"출력: {result}\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    test()
