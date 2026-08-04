import os
import re
from datasets import load_dataset
from tqdm import tqdm

def download_gsm8k_ko(output_file="train_data/gsm8k_ko.txt"):
    """
    HuggingFace의 kuotient/gsm8k-ko (수학 추론 데이터셋)을 다운로드하여
    질문(Q) - 단계별 풀이과정(A) 포맷으로 정제 후 저장.
    """
    print("⏳ GSM8K-ko 수학 추론 데이터셋 다운로드 시작...")
    
    try:
        # 데이터셋 로드
        dataset = load_dataset("kuotient/gsm8k-ko", split="train")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    formatted_data = []
    
    # 정규식으로 주석 제거 및 텍스트 정리
    for item in tqdm(dataset, desc="정제 중"):
        question = item['question'].strip().replace("\n", " ")
        answer = item['answer'].strip()
        
        # GSM8K 특유의 계산식 태그(<<10+5=15>>) 제거
        answer = re.sub(r'<<.*?>>', '', answer)
        
        # 줄바꿈 정제
        answer_lines = [line.strip() for line in answer.split('\n') if line.strip()]
        clean_answer = "\n".join(answer_lines)
        
        # Q: & A: 형식으로 구조화
        formatted_entry = f"Q: {question}\nA: {clean_answer}"
        formatted_data.append(formatted_entry)
        
    print(f"💾 추출 완료! [{output_file}] 에 저장 중...")
    with open(output_file, "w", encoding="utf-8") as f:
        # 각 문항 구분을 위해 두 줄 개행(\n\n) 처리
        f.write("\n\n".join(formatted_data) + "\n")
        
    print(f"✅ 저장 완료: {output_file} (총 {len(formatted_data)}개 수학 추론 문항)")

if __name__ == "__main__":
    download_gsm8k_ko()
