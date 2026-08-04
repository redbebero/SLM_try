import torch
from datasets import load_dataset
from tokenizer import KoJamoTokenizer
import re
from tqdm import tqdm

def prepare_clean_wiki(output_file="wiki_clean_short.txt", target_sentences=10000):
    print("⏳ 위키백과 데이터 스트리밍 시작 (wikimedia/wikipedia: 20231101.ko)...")
    
    # 디스크 용량을 낭비하지 않도록 streaming=True 사용
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    
    tokenizer = KoJamoTokenizer()
    unk_id = tokenizer.unk_token_id
    
    valid_sentences = []
    
    # 텍스트를 문장 단위로 쪼개기 위한 정규식 (마침표 뒤 띄어쓰기)
    sentence_splitter = re.compile(r'(?<=\.)\s+')
    
    print(f"🎯 목표: 순도 99.9% 한국어 단문 {target_sentences}개 수집 (엄격한 3중 필터 적용)")
    pbar = tqdm(total=target_sentences)
    
    for item in dataset:
        text = item['text']
        sentences = sentence_splitter.split(text)
        
        for sent in sentences:
            sent = sent.strip()
            
            # [필터 1] 길이 필터 (10자 이상 ~ 30자 이하의 단문만 허용)
            if not (10 <= len(sent) <= 30):
                continue
                
            # [필터 2] 노이즈 기호 원천 배제 (괄호, 따옴표, LaTeX 문법 기호 등)
            forbidden_chars = ['\\', '{', '}', '^', '_', '[', ']', '(', ')', '<', '>', '@', '#', '$', '%', '&', '*', '"', "'", '·']
            if any(c in sent for c in forbidden_chars):
                continue
                
            # [필터 3] 수학적 물리엔진 토크나이저 검증 (한자나 이모지 등 UNK 1개라도 있으면 폐기)
            encoded = tokenizer.encode(sent)
            extra_track = encoded[:, 3]
            
            if unk_id in extra_track:
                continue
                
            # 모든 혹독한 필터를 통과한 문장
            if not sent.endswith('.'):
                sent += '.'
                
            valid_sentences.append(sent)
            pbar.update(1)
            
            if len(valid_sentences) >= target_sentences:
                break
                
        if len(valid_sentences) >= target_sentences:
            break
            
    pbar.close()
    
    print(f"💾 추출 완료! [{output_file}] 에 저장 중...")
    with open(output_file, "w", encoding="utf-8") as f:
        for sent in valid_sentences:
            f.write(f"{sent}\n")
            
    print("✅ 데이터 정제 및 저장 완료!")
    
    # 통과된 샘플 몇 개 출력
    print("\n[샘플 확인]")
    for i in range(5):
        print(f" - {valid_sentences[i]}")

if __name__ == "__main__":
    prepare_clean_wiki("wiki_clean_short.txt", target_sentences=10000)
