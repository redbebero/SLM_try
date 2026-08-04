import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from tokenizer import KoJamoTokenizer
from chat import load_model, generate

def test():
    checkpoint = "checkpoints/model_sft_v53.pth"
    if not os.path.exists(checkpoint):
        print(f"❌ 체크포인트 없음: {checkpoint}")
        return
        
    tokenizer = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()
    
    print(f"⏳ 로딩 중: {checkpoint}")
    model, device = load_model(checkpoint, vocab_sizes)
    
    test_cases = [
        ("안녕! 너는 누구야?", 50, "기본 인사"),
        ("오늘 날씨 너무 좋다. 너는 오늘 뭐 하고 놀 거야?", 60, "일상 대화"),
        ("임진왜란이 몇 년도에 일어났어?", 60, "지식 기권 1"),
        ("프랑스의 수도가 어디인지 혹시 알아?", 60, "지식 기권 2"),
        ("조선 태조 이성계는 고려를 멸망시키고 조선을 건국한 인물이다. 조선을 건국한 인물은 누구인가?", 80, "신규 지문 독해")
    ]
    
    for q, max_chars, title in test_cases:
        prompt = f"Q: {q}\nA: "
        print(f"\n[{title}]")
        print(f"Q: {q}")
        result = generate(model, tokenizer, prompt, max_new_chars=max_chars, device=device, stop_on_newline=True)
        print(f"🤖 A: {result}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    test()
