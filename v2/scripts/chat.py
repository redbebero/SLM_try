import torch
import sys
import os
import glob
import re

# v2 폴더 기준 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from tokenizer import KoJamoTokenizer
from config import EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT


def resolve_checkpoint(version=None):
    """SFT 모델(model_sft_v*.pth)이 있으면 우선 로드, 없으면 일반 model_v*.pth 로드."""
    if version is not None:
        sft_path = f"checkpoints/model_sft_v{version}.pth"
        if os.path.exists(sft_path):
            return sft_path
        return f"checkpoints/model_v{version}.pth"
        
    sft_files = glob.glob("checkpoints/model_sft_v*.pth")
    if sft_files:
        nums = [(int(re.search(r"model_sft_v(\d+)\.pth", f).group(1)), f)
                for f in sft_files if re.search(r"model_sft_v(\d+)\.pth", f)]
        return max(nums, key=lambda x: x[0])[1]
        
    files = glob.glob("checkpoints/model_v*.pth")
    nums  = [(int(re.search(r"model_v(\d+)\.pth", f).group(1)), f)
             for f in files if re.search(r"model_v(\d+)\.pth", f)]
    return max(nums, key=lambda x: x[0])[1] if nums else None


def load_model(checkpoint_path, vocab_sizes, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KoJamoNet(vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
                      num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 신규 포맷(model/optimizer/scheduler/epoch 딕셔너리)과 구 포맷(순수 state_dict) 모두 지원
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def generate(model, tokenizer, prompt, max_new_chars=50, device="cpu", stop_on_newline=False):
    """프롬프트를 받아 텍스트를 이어서 생성합니다.
    stop_on_newline=True면 개행 생성 시 그 자리에서 멈춤 (SFT 모델의 턴 경계 인식용)."""
    current_seq = tokenizer.encode(prompt).unsqueeze(0).to(device)
    gen_count = 0
    newline_id = tokenizer.sym_vocab.get('\n')

    with torch.no_grad():
        for _ in range(max_new_chars):
            # 문맥이 1,000자를 넘길 경우 가장 오래된 텍스트는 버리고 최신 1,000자만 유지 (메모리 보전)
            if current_seq.shape[1] > 1000:
                current_seq = current_seq[:, -1000:, :]

            # model 아웃풋에 logits_type 추가 수신
            t_logits, c, ju, jo, sym, eng, num = model(current_seq)
            
            # 다음 시점의 문자 타입 예측 (0: 한국어, 1: 기호, 2: 영어, 3: 숫자)
            pred_type = t_logits[:, -1, :].argmax(dim=-1).item()

            # 기본적으로 모든 트랙 0(PAD)으로 초기화 (1, 1) 2D 텐서
            next_cho  = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_jung = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_jong = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_sym  = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_eng  = torch.zeros(1, 1, dtype=torch.long, device=device)
            next_num  = torch.zeros(1, 1, dtype=torch.long, device=device)

            # 예측된 타입에 맞는 트랙의 예측값만 활성화
            if pred_type == 0:  # 한국어
                next_cho  = c[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                next_jung = ju[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                next_jong = jo[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            elif pred_type == 1:  # 기호
                next_sym  = sym[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            elif pred_type == 2:  # 영어
                next_eng  = eng[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            elif pred_type == 3:  # 숫자
                next_num  = num[:, -1, :].argmax(dim=-1).unsqueeze(-1)

            # (1, 6) 2D 생성 후 unsqueeze(1)을 통해 (1, 1, 6) 3D 텐서로 확장하여 결합
            next_token = torch.cat([next_cho, next_jung, next_jong, next_sym, next_eng, next_num], dim=-1).unsqueeze(1)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            gen_count += 1

            # 턴 종료 감지: 개행이 나오면 답변이 끝난 것으로 보고 멈춤 (최소 1글자는 생성한 후)
            if stop_on_newline and gen_count > 1 and pred_type == 1 and next_sym.item() == newline_id:
                break

    return tokenizer.decode(current_seq[0, -gen_count:]).rstrip("\n")


def main():
    print("=" * 50)
    print("  🧠 Ko-JamoNet v2 — 6-Track GRU 모델")
    print("  (종료: 'q' 또는 'quit' 입력)")
    print("=" * 50)

    version    = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    checkpoint = resolve_checkpoint(version)

    if checkpoint is None or not os.path.exists(checkpoint):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint}" if checkpoint
              else "❌ checkpoints/ 에 model_v*.pth 파일이 없습니다.")
        print("   먼저 train.py를 실행해 모델을 학습시켜 주세요.")
        return

    tokenizer   = KoJamoTokenizer()
    vocab_sizes = tokenizer.get_vocab_sizes()

    # SFT(Q:/A: 턴 구조로 학습된) 모델인지 여부에 따라 프롬프트 템플릿/멈춤조건을 다르게 적용
    is_sft = "model_sft_v" in os.path.basename(checkpoint)

    print(f"⏳ 모델 로딩 중... ({checkpoint})")
    model, device = load_model(checkpoint, vocab_sizes)
    print(f"✅ 모델 로딩 완료! (Device: {device}, SFT모드: {is_sft})\n")

    while True:
        try:
            user_input = input("📝 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("👋 종료합니다.")
            break

        if is_sft:
            # 학습 데이터와 동일한 "Q: ...\nA: " 템플릿으로 감싸고, 개행에서 멈춤
            prompt = f"Q: {user_input}\nA: "
            result = generate(model, tokenizer, prompt, max_new_chars=200, device=device, stop_on_newline=True)
        else:
            result = generate(model, tokenizer, user_input, max_new_chars=50, device=device)
        print(f"🤖 출력> {result}\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts 폴더 안에 있는 경우 상위 v2 루트 폴더로 작업 경로 이동하여 경로 일관성 유지
    parent_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    os.chdir(parent_dir)
    main()
