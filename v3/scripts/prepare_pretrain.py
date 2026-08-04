import os

PROJECT_DIR = "/home/redbebero/Projects/SLM/v2"
wiki_path = os.path.join(PROJECT_DIR, "datasets", "wiki_clean_short.txt")
persona_path = os.path.join(PROJECT_DIR, "datasets", "persona_data.txt")
combined_path = os.path.join(PROJECT_DIR, "datasets", "pretrain_combined.txt")

print("⏳ 프리트레인용 데이터셋 통합 중...")

combined_lines = []

# 1. 위키백과 데이터 추가
if os.path.exists(wiki_path):
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki_lines = f.readlines()
        print(f"- 위키백과 데이터 로드 완료: {len(wiki_lines)}개 문장")
        for line in wiki_lines:
            clean = line.strip()
            if clean:
                combined_lines.append(clean)
else:
    print(f"⚠️ 위키백과 파일이 없습니다: {wiki_path}")

# 2. 페르소나 데이터에서 Q:, A: 제거 후 순수 대화 문장만 추가
if os.path.exists(persona_path):
    persona_count = 0
    with open(persona_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            # Q: 또는 A: 접두사 제거
            if clean.startswith("Q: "):
                clean = clean[3:]
            elif clean.startswith("A: "):
                clean = clean[3:]
            
            combined_lines.append(clean)
            persona_count += 1
    print(f"- 페르소나 대화 데이터 로드 완료 (접두사 제거): {persona_count}개 문장")
else:
    print(f"⚠️ 페르소나 파일이 없습니다: {persona_path}")

# 3. 통합 파일 저장
with open(combined_path, "w", encoding="utf-8") as f:
    f.write("\n".join(combined_lines) + "\n")

print(f"✅ 프리트레인 통합 파일 생성 완료: {combined_path} (총 {len(combined_lines)}개 문장)")
