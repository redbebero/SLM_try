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

# ── Phase 2: 실제 인간 질문 기반 지식 기권 (Zero-Template Refusal) ─────────────
def generate_refusal_samples_from_korquad(limit=300):
    json_path = "datasets/KorQuAD_v1.0_train.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_questions = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                question = clean_text(qa["question"])
                if 10 < len(question) < 50:
                    raw_questions.append(question)
                    
    random.shuffle(raw_questions)
    selected_questions = raw_questions[:limit]
    
    samples = []
    for q in selected_questions:
        politeness = random.choice(["polite", "casual"])
        if politeness == "polite":
            a = random.choice([
                "저는 배경지식이 없어 알지 못합니다. 하지만 관련된 지문을 주시면 정답을 찾아드릴게요!",
                "죄송해요, 저는 아는 지식이 없어요. 정보가 담긴 지문을 같이 주시면 읽고 답변해 드릴 수 있어요.",
                "제 머릿속에는 상식이 들어있지 않아요. 지문을 보여주시면 그 안에서 답을 찾아낼게요.",
                "그 지식은 제게 없답니다. 하지만 질문 내용이 포함된 지문을 주시면 정확히 짚어드릴게요!"
            ])
        else:
            a = random.choice([
                "난 배경지식이 없어 몰라. 하지만 관련 지문을 주면 정답을 찾아줄게!",
                "미안해, 난 아는 지식이 없어. 정보가 담긴 글을 보여주면 읽고 답해줄 수 있어.",
                "내 머리엔 상식이 안 들어있어. 지문 보여주면 그 안에서 정답 짚어줄게.",
                "그 지식은 나한테 없어. 그렇지만 질문 내용이 있는 지문을 주면 알려줄게!"
            ])
        samples.append(f"Q: {q}\nA: {a}")
    return samples

# ── Phase 4: KorQuAD & GSM8K & Persona 파싱 (역할극 전량 삭제) ────────────────
def load_korquad_samples(limit=500):
    json_path = "datasets/KorQuAD_v1.0_train.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = clean_text(paragraph["context"])
            if len(context) > 350 or len(context) < 100:
                continue
            for qa in paragraph["qas"]:
                question = clean_text(qa["question"])
                if len(question) > 65 or len(question) < 7:
                    continue
                if not qa["answers"]:
                    continue
                answer = clean_text(qa["answers"][0]["text"])
                if len(answer) > 40 or len(answer) == 0:
                    continue
                if context.count(answer) != 1:
                    continue
                evidence = find_evidence_sentence(context, answer)
                if not evidence:
                    continue
                    
                formatted = (
                    f"Q: {context} {question}\n"
                    f"A: 지문에서 \"{evidence}\"라고 언급한 부분에 근거하여, 질문의 답은 \"{answer}\"입니다."
                )
                samples.append(formatted)
                if len(samples) >= limit:
                    return samples
    return samples

def load_gsm8k_samples(limit=200):
    path = "datasets/gsm8k_ko.txt"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    raw_blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    samples = []
    for b in raw_blocks:
        if "Q:" in b and "A:" in b:
            if len(b) < 350:
                samples.append(b)
                if len(samples) >= limit:
                    return samples
    return samples

def load_persona_samples(limit=1300): # 일상 카톡 대화를 1300개로 대폭 확장
    path = "datasets/persona_data.txt"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    raw_blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    samples = []
    for b in raw_blocks:
        lines = b.split("\n")
        if len(lines) >= 2 and lines[0].startswith("Q: ") and lines[1].startswith("A: "):
            q = lines[0].replace("Q: ", "").strip()
            a = lines[1].replace("A: ", "").strip()
            if len(q) < 50 and len(a) < 50:
                samples.append(f"Q: {q}\nA: {a}")
                if len(samples) >= limit:
                    return samples
    return samples

# ── Phase 5: 최종 컴파일 및 믹싱 ────────────────────────────────
def build_hybrid_dataset():
    print("⏳ [Phase 4] 원천 데이터 파싱 및 설명형 가공 시작...")
    korquad_data = load_korquad_samples(limit=500)
    gsm8k_data = load_gsm8k_samples(limit=200)
    persona_data = load_persona_samples(limit=1300)
    
    print("⏳ [Phase 2] 인간 질문 기반 지식 기권/독해 유도 합성 시작...")
    refusal_data = generate_refusal_samples_from_korquad(limit=300)
    
    print(f"  └─ 독해: {len(korquad_data)}개 | 연산: {len(gsm8k_data)}개")
    print(f"  └─ 순수 일상 대화(페르소나): {len(persona_data)}개")
    print(f"  └─ 기권(실제 상식 질문): {len(refusal_data)}개")
    
    final_dataset = korquad_data + gsm8k_data + persona_data + refusal_data
    random.shuffle(final_dataset)
    
    sft_dir = "train_data_sft"
    os.makedirs(sft_dir, exist_ok=True)
    out_path = os.path.join(sft_dir, "korquad_sft.txt")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(final_dataset))
        
    print(f"\n✅ [Phase 5] 최종 순수대화형 SFT 데이터셋 구축 완료 -> {out_path} (총 {len(final_dataset)}개 샘플)")

if __name__ == "__main__":
    build_hybrid_dataset()
