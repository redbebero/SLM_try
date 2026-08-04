import os
import re
import sys
from datasets import load_dataset
from tqdm import tqdm

# v2 폴더 기준 임포트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import KoJamoTokenizer

def prepare_paragraphs(output_file="train_data/wiki_paragraphs.txt", target_paragraphs=15000):
    print("⏳ 위키백과 데이터 스트리밍 시작 (wikimedia/wikipedia: 20231101.ko)...")
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 데이터셋 스트리밍 로드 (로컬 캐시 활용)
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    tokenizer = KoJamoTokenizer()
    unk_id = tokenizer.unk_token_id
    
    valid_paragraphs = []
    
    # 클리닝 정규식 (대괄호 주석 [1], [2] 등 제거, 둥근 괄호 내부 한자/외래어 제거)
    citation_pattern = re.compile(r'\[\d+\]')
    parenthesis_pattern = re.compile(r'\([^)]*\)')
    whitespace_pattern = re.compile(r'\s+')
    
    print(f"🎯 목표: 논리적 인과관계 학습용 고품질 문단 {target_paragraphs}개 수집")
    pbar = tqdm(total=target_paragraphs)
    
    for item in dataset:
        text = item['text']
        
        # 문단 단위 분할 (\n\n 또는 \n)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for para in paragraphs:
            # 1. 괄호 안의 외래어/한자 및 주석 제거
            para = citation_pattern.sub('', para)
            para = parenthesis_pattern.sub('', para)
            para = whitespace_pattern.sub(' ', para).strip()
            
            # 2. 길이 필터 (150자 이상 ~ 800자 이하의 적당한 문단만 허용)
            # 수용야 1000자 안에서 완전한 문맥(서론-본론-결론)을 학습할 수 있게 설계
            if not (150 <= len(para) <= 800):
                continue
                
            # 3. 노이즈 및 유해 기호 검사 (수식 기호 등 제거)
            forbidden_chars = ['\\', '{', '}', '^', '_', '<', '>', '@', '#', '$', '%', '&', '*']
            if any(c in para for c in forbidden_chars):
                continue
                
            # 4. UNK 토큰 비율 검사 (한자/이모지가 너무 많으면 배제)
            encoded = tokenizer.encode(para)
            extra_track = encoded[:, 3]
            unk_count = (extra_track == unk_id).sum().item()
            unk_ratio = unk_count / len(para)
            
            # UNK 글자 비율이 3% 이상이면 탈락
            if unk_ratio > 0.03:
                continue
                
            valid_paragraphs.append(para)
            pbar.update(1)
            
            if len(valid_paragraphs) >= target_paragraphs:
                break
                
        if len(valid_paragraphs) >= target_paragraphs:
            break
            
    pbar.close()
    
    print(f"💾 추출 완료! [{output_file}] 에 저장 중...")
    with open(output_file, "w", encoding="utf-8") as f:
        for para in valid_paragraphs:
            # 문단 구분을 확실히 하기 위해 두 개의 줄바꿈(\n\n)으로 저장
            f.write(f"{para}\n\n")
            
    print("✅ 데이터 정제 및 저장 완료!")
    
    # 샘플 출력
    print("\n[추출 문단 샘플 확인]")
    for i in range(min(3, len(valid_paragraphs))):
        print(f"[{i+1}] {valid_paragraphs[i][:150]}...")
        print("-" * 50)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "train_data/wiki_paragraphs.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    prepare_paragraphs(output_file=out, target_paragraphs=n)
