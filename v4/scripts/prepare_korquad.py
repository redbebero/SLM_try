import json
import urllib.request
import os
import re
import random

def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'[^\w\s.,?!~₩%@#\-+\'\"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_evidence_sentence(context, answer):
    sentences = re.split(r'(?<=[.!?])\s+', context)
    for s in sentences:
        if answer in s:
            return s.strip()
    return ""

def generate_refusal_samples():
    # CFG 기반 기권 및 독해 유도 (Symbolic Rule Builder)
    topics = [
        "지구", "태양", "블랙홀", "광합성", "이순신", "세종대왕", "조선", "임진왜란", "미국 수도", "프랑스 수도",
        "파이썬", "양자역학", "피타고라스 공식", "인공지능", "컴퓨터", "휴대폰 가격", "내일 날씨", "세계 인구",
        "가장 높은 산", "주식 시장", "비트코인", "독도 주소", "올림픽 역사", "도서관 위치", "외계인", "공룡",
        "중력 법칙", "광속", "상대성 이론", "인간의 뇌", "심리학", "법률 상담", "의학 지식", "우주선 작동 원리"
    ]
    prefixes = ["혹시", "너", "나한테", "갑자기 궁금한데", "도와줘,"]
    verbs = ["알려줄 수 있어?", "설명해봐", "아는지 물어볼게.", "가르쳐줘.", "뭔지 알아?"]
    
    refusal_templates = [
        "저는 외부 지식이 들어있지 않습니다. 하지만 내용을 적은 지문을 주시면 읽고 답해 드릴 수 있어요!",
        "죄송하지만 제 머릿속에는 지식이 없어요. 정보가 담긴 글(지문)을 같이 첨부해 주시면 바로 찾아낼게요.",
        "배경지식이 없어 알지 못하는 내용입니다. 질문 내용이 실린 지문을 보여주시면 족집게처럼 정답을 짚어드릴게요!"
    ]
    
    samples = []
    for _ in range(400):
        t = random.choice(topics)
        p = random.choice(prefixes)
        v = random.choice(verbs)
        
        q = f"{p} {t} {v}"
        a = random.choice(refusal_templates)
        samples.append(f"Q: {q}\nA: {a}")
        
    random.shuffle(samples)
    return samples[:400]

def generate_chitchat_samples():
    # CFG 기반 일상형 공감 대화 (Chitchat)
    greetings = ["안녕하세요", "안녕", "반가워요", "반갑습니다", "하이", "안녕 친구", "좋은 하루예요", "반갑구만"]
    gr_answers = [
        "안녕하세요! 저는 지식을 읽어주는 꼬마 비서입니다.",
        "반가워요! 어떤 이야기를 나눌까요?",
        "반갑습니다! 저는 5살 독해 로봇이에요."
    ]
    
    openings = ["오늘", "요즘", "갑자기", "왠지 모르게", "참", "오늘따라"]
    subjects = ["기분이", "마음이", "하루가", "몸 상태가", "머리가", "온몸이"]
    states = [
        ("슬퍼서", "슬프군요. 토닥토닥 위로해 드릴게요."),
        ("우울해서 눈물 나", "눈물이 나면 참지 마세요. 제가 곁에서 토닥여 드릴게요."),
        ("기쁘고 신나", "와! 정말 기쁜 소식이네요. 저도 행복해져요!"),
        ("피곤하고 지쳐", "오늘 하루 힘드셨군요. 푹 쉬면서 제 응원을 받아보세요."),
        ("심심하고 지루해", "저랑 재미있는 대화를 나누며 심심함을 풀어볼까요?"),
        ("행복해서 웃음이 나", "미소 짓는 얼굴이 상상돼요. 계속 행복하시길 바랄게요!"),
        ("화가 나고 답답해", "후우, 정말 짜증나고 답답하셨겠어요. 무슨 일인지 제게 말해주세요."),
        ("불안하고 외로워", "불안해하지 마세요. 제가 항상 여기서 친구가 되어 드릴게요.")
    ]
    endings = ["", " 힘드네.", " 어떡하지?", " 진짜 그렇네."]
    
    samples = []
    for _ in range(450):
        o = random.choice(openings)
        s = random.choice(subjects)
        state_word, state_ans = random.choice(states)
        e = random.choice(endings)
        
        q = f"{o} {s} {state_word}{e}"
        a = state_ans
        samples.append(f"Q: {q}\nA: {a}")
        
    for g in greetings:
        for a in gr_answers:
            samples.append(f"Q: {g}\nA: {a}")
            
    random.shuffle(samples)
    return samples[:500]

def download_and_parse_korquad():
    url = "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json"
    dest_dir = "datasets"
    os.makedirs(dest_dir, exist_ok=True)
    json_path = os.path.join(dest_dir, "KorQuAD_v1.0_train.json")
    
    if not os.path.exists(json_path):
        print(f"📥 KorQuAD 1.0 다운로드 중: {url} ...")
        urllib.request.urlretrieve(url, json_path)
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("⚙️ 독해 데이터 가공 및 필터링...")
    qa_samples = []
    
    for article in data["data"]:
        article_qa_count = 0
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            cleaned_context = clean_text(context)
            
            if len(cleaned_context) > 350 or len(cleaned_context) < 100:
                continue
                
            for qa in paragraph["qas"]:
                if article_qa_count >= 4:
                    break
                question = clean_text(qa["question"])
                if len(question) > 65 or len(question) < 7:
                    continue
                if not qa["answers"]:
                    continue
                answer = clean_text(qa["answers"][0]["text"])
                if len(answer) > 40 or len(answer) == 0:
                    continue
                if cleaned_context.count(answer) != 1:
                    continue
                
                # 근거 문장
                evidence_sentence = find_evidence_sentence(cleaned_context, answer)
                if not evidence_sentence:
                    continue
                
                tot_len = len(cleaned_context) + len(question) + len(evidence_sentence) + len(answer)
                if tot_len > 430:
                    continue
                    
                formatted = (
                    f"Q: 지문: {cleaned_context} 질문: {question}\n"
                    f"A: 지문에서 \"{evidence_sentence}\"라고 언급한 부분에 근거하여, 질문의 답은 \"{answer}\"입니다."
                )
                qa_samples.append(formatted)
                article_qa_count += 1
                
                if len(qa_samples) >= 600: # 독해는 600개로 균형 유지
                    break
            if len(qa_samples) >= 600:
                break
        if len(qa_samples) >= 600:
            break

    # 일상 대화 및 기권 데이터 생성
    chat_samples = generate_chitchat_samples()
    refusal_samples = generate_refusal_samples()
    
    print(f"💬 일상 대화: {len(chat_samples)}개 | 기권 유도: {len(refusal_samples)}개 | 독해: {len(qa_samples)}개")
    
    # 3대 그룹 혼합
    final_samples = qa_samples + chat_samples + refusal_samples
    random.shuffle(final_samples)
    
    # 디렉토리 저장
    sft_dir = "train_data_sft"
    os.makedirs(sft_dir, exist_ok=True)
    out_path = os.path.join(sft_dir, "korquad_sft.txt")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(final_samples))
        
    print(f"💾 SFT 최종 하이브리드 대화 데이터셋 저장 완료 -> {out_path} ({len(final_samples)}개 샘플)")

if __name__ == "__main__":
    download_and_parse_korquad()
