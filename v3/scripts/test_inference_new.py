import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from tokenizer import KoJamoTokenizer
from chat import load_model, generate

def test():
    checkpoint = "checkpoints/model_sft_v48.pth"
    if not os.path.exists(checkpoint):
        print(f"❌ 체크포인트 없음: {checkpoint}")
        return
        
    tokenizer = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()
    
    print(f"⏳ 로딩 중: {checkpoint}")
    model, device = load_model(checkpoint, vocab_sizes)
    
    # 1. 일상 스몰토크 테스트 (완전히 새로운 일상 구문)
    prompt1 = "Q: 오늘 날씨 너무 좋다. 너는 주말에 보통 뭐 하고 보내?\nA: "
    print(f"\n[Test 1 - 일상 스몰토크]")
    print(f"Q: 오늘 날씨 너무 좋다. 너는 주말에 보통 뭐 하고 보내?")
    result1 = generate(model, tokenizer, prompt1, max_new_chars=60, device=device, stop_on_newline=True)
    print(f"🤖 A: {result1}")
    
    # 2. 지식 기권 테스트 (본문 없는 상식 질문)
    prompt2 = "Q: 지구에서 달까지의 거리가 정확히 몇 킬로미터야?\nA: "
    print(f"\n[Test 2 - 지식 기권]")
    print(f"Q: 지구에서 달까지의 거리가 정확히 몇 킬로미터야?")
    result2 = generate(model, tokenizer, prompt2, max_new_chars=60, device=device, stop_on_newline=True)
    print(f"🤖 A: {result2}")
    
    # 3. 신규 지문 독해 테스트 (한 줄 결합형 독해)
    context = "홍길동은 조선 중기의 뛰어난 의적이다. 그는 활빈당이라는 백성 구율 단체를 조직해 부패한 관료들의 재물을 빼앗아 가난한 사람들을 지원했다."
    question = "홍길동이 백성을 돕기 위해 조직한 단체의 이름은 무엇인가?"
    prompt3 = f"Q: {context} {question}\nA: "
    
    print(f"\n[Test 3 - 신규 지문 독해]")
    print(f"Q: {context} {question}")
    result3 = generate(model, tokenizer, prompt3, max_new_chars=120, device=device, stop_on_newline=True)
    print(f"🤖 A: {result3}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    test()
