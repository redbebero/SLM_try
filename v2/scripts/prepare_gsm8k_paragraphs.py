import os
import re

def prepare_gsm8k_paragraphs(input_file="datasets/gsm8k_ko.txt", output_file="train_data/gsm8k_paragraphs.txt"):
    """
    SFT용 gsm8k_ko.txt 데이터셋에서 Q: 와 A: 및 #### 정답 구분선을 제거하고
    순수한 수학적 문제-풀이 줄글 문단으로 가공하여 pretrain 데이터셋으로 주입.
    이를 통해 pretrain 단계에서 문답 포맷 오염 없이 '수학적 인과관계 추론'을 선습득하도록 함.
    """
    print(f"⏳ 수학 추론 데이터 Pretrain용 가공 시작: {input_file} -> {output_file}")
    
    if not os.path.exists(input_file):
        # 혹시 train_data 폴더에 남아있을 경우 예외 처리
        fallback = "train_data/gsm8k_ko.txt"
        if os.path.exists(fallback):
            input_file = fallback
        else:
            print(f"❌ 원본 파일이 없습니다: {input_file}")
            return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    entries = [e.strip() for e in content.split("\n\n") if e.strip()]
    cleaned_paragraphs = []
    
    for entry in entries:
        # Q: 및 A: 태그 제거
        lines = entry.split("\n")
        clean_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            
            # 접두사 및 정답 표시 제거
            if line_str.startswith("Q: "):
                line_str = line_str[3:]
            elif line_str.startswith("A: "):
                line_str = line_str[3:]
            elif line_str.startswith("#### "):
                continue
                
            clean_lines.append(line_str)
            
        if clean_lines:
            # 하나의 연속된 문제-풀이 문단으로 병합
            paragraph = " ".join(clean_lines)
            cleaned_paragraphs.append(paragraph)
            
    print(f"📝 정제 완료: {len(cleaned_paragraphs)}개 문항 줄글 변환 완료")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # 두 번 개행으로 문단 구분
        f.write("\n\n".join(cleaned_paragraphs) + "\n\n")
        
    print(f"✅ 저장 완료: {output_file}")

if __name__ == "__main__":
    prepare_gsm8k_paragraphs()
