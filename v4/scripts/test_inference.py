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
    # 최신 SFT 가중치 강제 지정
    checkpoint = "checkpoints/model_sft_v124.pth"
    if not os.path.exists(checkpoint):
        pytest.skip(f"optional legacy checkpoint missing: {checkpoint}")
    tokenizer = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()
    
    print(f"⏳ 로딩 중: {checkpoint}")
    model, device = load_model(checkpoint, vocab_sizes)
    
    # 평가 샘플 1 (역사 인물 추출)
    context1 = "리슐리외는 프랑스 루이 13세 시기의 총리이다. 그는 중앙집권체제의 확립과 왕권 강화에 힘을 쏟았으며 절대왕정의 기초를 쌓았다."
    question1 = "절대왕정의 기초를 쌓은 인물은 누구인가?"
    prompt1 = f"Q: 지문: {context1} 질문: {question1}\nA: "
    
    print(f"\n[평가 1]")
    print(f"프롬프트: {prompt1}")
    result1 = generate(model, tokenizer, prompt1, max_new_chars=50, device=device, stop_on_newline=True)
    print(f"🤖 답변: {result1}")
    
    # 평가 샘플 2 (연도 숫자 추출)
    context2 = "원더걸스는 2007년에 데뷔하여 활발히 활동하였다. 2009년에는 미국에서 Nobody 싱글을 발매했다."
    question2 = "원더걸스가 미국에서 Nobody를 발매한 연도는?"
    prompt2 = f"Q: 지문: {context2} 질문: {question2}\nA: "
    
    print(f"\n[평가 2]")
    print(f"프롬프트: {prompt2}")
    result2 = generate(model, tokenizer, prompt2, max_new_chars=50, device=device, stop_on_newline=True)
    print(f"🤖 답변: {result2}")

if __name__ == "__main__":
    # scripts 폴더 안에 있는 경우 상위 v3 루트 폴더로 작업 경로 이동
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    test()
