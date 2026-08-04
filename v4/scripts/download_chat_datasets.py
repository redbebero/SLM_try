import sys
import os
from datasets import load_dataset

# v2 폴더 경로
PROJECT_DIR = "/home/redbebero/Projects/SLM/v2"
os.makedirs(PROJECT_DIR, exist_ok=True)

roleplay_path = os.path.join(PROJECT_DIR, "datasets", "roleplay_data.txt")
persona_path = os.path.join(PROJECT_DIR, "datasets", "persona_data.txt")

print("⏳ 1. korean-role-playing 다운로드 및 처리 중...")
try:
    ds1 = load_dataset("huggingface-KREW/korean-role-playing", "general-roleplay-data", split="train")
    
    with open(roleplay_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds1):
            dialog_list = item["text"]
            formatted_dialog = []
            for turn in dialog_list:
                role = "Q" if turn["role"] == "user" else "A"
                content = turn["content"].strip().replace("\n", " ")
                formatted_dialog.append(f"{role}: {content}")
            
            # 문장 구분을 위해 줄바꿈으로 합치고 대화 단위는 빈 줄로 분리
            f.write("\n".join(formatted_dialog) + "\n\n")
            
            # 첫 2개 대화만 콘솔 출력용으로 저장
            if idx < 2:
                print(f"\n[대화 {idx+1}]")
                print("\n".join(formatted_dialog))
                print("-" * 50)
                
    print(f"✅ 저장 완료: {roleplay_path} (총 {len(ds1)}개 대화)")
except Exception as e:
    print("ds1 처리 에러:", e)

print("\n⏳ 2. korean-persona-chat-dataset-v2 다운로드 및 처리 중...")
try:
    ds2 = load_dataset("NLPBada/korean-persona-chat-dataset-v2", split="train")
    
    with open(persona_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds2):
            dialog_list = item["session_dialog"]
            if isinstance(dialog_list, str):
                import ast
                try:
                    dialog_list = ast.literal_eval(dialog_list)
                except:
                    pass
            
            formatted_dialog = []
            for t_idx, turn in enumerate(dialog_list):
                role = "Q" if t_idx % 2 == 0 else "A"
                content = turn.strip().replace("\n", " ")
                formatted_dialog.append(f"{role}: {content}")
                
            f.write("\n".join(formatted_dialog) + "\n\n")
            
            if idx < 2:
                print(f"\n[대화 {idx+1}]")
                print("\n".join(formatted_dialog))
                print("-" * 50)
                
    print(f"✅ 저장 완료: {persona_path} (총 {len(ds2)}개 대화)")
except Exception as e:
    print("ds2 처리 에러:", e)
