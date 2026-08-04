import os
import re

def prepare_roleplay(input_file="datasets/roleplay_data.txt", output_file="train_data/roleplay_paragraphs.txt", sample_ratio=0.3):
    """
    역할놀이 데이터(167MB)를 가공하여 pretrain용 파일로 저장.
    사용자의 요청에 따라 Q: 와 A: 접두사를 그대로 보존하여
    모델이 프롬프트와 답변 간의 인과적 대화 관계를 함께 학습할 수 있도록 함.
    """
    print(f"⏳ 역할놀이 대화형 데이터 정제 시작 (Q&A 태그 보존): {input_file} -> {output_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ 원본 파일이 없습니다: {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 대화 세션 단위로 나누기 위해 파일 전체를 읽어 빈 줄로 분할
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    sessions = [s.strip() for s in content.split("\n\n") if s.strip()]
    
    cleaned_sessions = []
    forbidden_chars = ['\\', '{', '}', '^', '_', '<', '>']
    
    for session in sessions:
        # 노이즈 기호가 섞인 세션은 필터링
        if any(c in session for c in forbidden_chars):
            continue
            
        # 개행 구조 유지하면서 정제
        lines = [line.strip() for line in session.split('\n') if line.strip()]
        if lines:
            cleaned_sessions.append("\n".join(lines))

    # 샘플링 (VRAM 및 학습 시간 관리)
    total_len = len(cleaned_sessions)
    keep_len = int(total_len * sample_ratio)
    
    step = max(1, total_len // keep_len) if keep_len > 0 else 1
    sampled_sessions = cleaned_sessions[::step][:keep_len]
    
    print(f"📝 정제 완료: 원본 {total_len} 세션 -> {len(sampled_sessions)} 세션 샘플링 (Q&A 형식 유지)")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # 각 대화 세션을 빈 줄(두 번 개행)로 연결하여 저장
        f.write("\n\n".join(sampled_sessions) + "\n\n")
            
    print(f"✅ 저장 완료: {output_file}")

if __name__ == "__main__":
    prepare_roleplay()
