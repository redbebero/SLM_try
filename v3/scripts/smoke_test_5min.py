"""
5분 smoke test: 위키 데이터 추출 → 정합성 검사 → 학습 → 평가
새 pretrain 방향이 동작하는지 빠르게 검증.
"""
import os, sys, time, re, math, signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import KoJamoTokenizer
from hrm_model import HRMContextNet

TRAIN_LIMIT_SEC = 5 * 60          # 학습 최대 5분
WIKI_SAMPLE     = 3000             # 위키 문단 수 (빠른 추출)
SEQ_LEN         = 128
BATCH           = 16
LR              = 3e-4
HIDDEN          = 128
EMB             = 16
CONTEXT_LAYERS  = 1
CONTEXT_HEADS   = 4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FILE = "train_data_hrm_pretrain_wiki/wiki_smoke.txt"
CKPT      = "checkpoints/smoke_5min_best.pth"

# ──────────────────────────────────────────────
# STEP 1: 위키 데이터 소량 추출
# ──────────────────────────────────────────────
def extract_wiki(output_file, target=WIKI_SAMPLE):
    if os.path.exists(output_file):
        lines = open(output_file, encoding="utf-8").read().strip().splitlines()
        clean = [l for l in lines if l.strip()]
        if len(clean) >= target // 2:
            print(f"[STEP1] 기존 파일 재사용: {output_file} ({len(clean)}줄)")
            return True

    print(f"[STEP1] 위키백과 스트리밍 추출 중 (목표 {target}개)...")
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("  ❌ datasets/tqdm 없음. pip install datasets tqdm")
        return False

    tokenizer = KoJamoTokenizer()
    unk_id = tokenizer.unk_token_id
    citation = re.compile(r'\[\d+\]')
    paren    = re.compile(r'\([^)]*\)')
    ws       = re.compile(r'\s+')
    forbidden = set('\\{}^_<>@#$%&*')

    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    except Exception as e:
        print(f"  ❌ 데이터셋 로드 실패: {e}")
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    collected = []
    pbar = tqdm(total=target, desc="위키 추출")
    for item in ds:
        for para in item["text"].split("\n"):
            para = citation.sub("", para)
            para = paren.sub("", para)
            para = ws.sub(" ", para).strip()
            if not (80 <= len(para) <= 500): continue
            if any(c in para for c in forbidden): continue
            enc = tokenizer.encode(para)
            if (enc[:, 3] == unk_id).sum().item() / len(para) > 0.02: continue
            collected.append(para)
            pbar.update(1)
            if len(collected) >= target: break
        if len(collected) >= target: break
    pbar.close()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(collected))
    print(f"  ✅ {len(collected)}개 추출 완료 → {output_file}")
    return True

# ──────────────────────────────────────────────
# STEP 2: 정합성 검사 (tokenizer round-trip)
# ──────────────────────────────────────────────
def validate_alignment(data_file):
    print("[STEP2] 토크나이저 정합성 검사 중...")
    tokenizer = KoJamoTokenizer()
    lines = open(data_file, encoding="utf-8").read().strip().splitlines()
    lines = [l for l in lines if l.strip()]

    errors = 0
    check_n = min(200, len(lines))
    for line in lines[:check_n]:
        enc = tokenizer.encode(line)
        dec = tokenizer.decode(enc)
        # UNK(?), 밑줄(_) 제외하고 비교
        orig_clean = line.replace("?", "").replace("_", "")
        dec_clean  = dec.replace("?", "").replace("_", "")
        if orig_clean != dec_clean:
            errors += 1

    ratio = errors / check_n
    print(f"  샘플 {check_n}개 검사: 오류 {errors}개 ({ratio*100:.1f}%)")
    if ratio < 0.01:
        print("  ✅ 정합성 통과 (오류율 < 1%)")
        return True
    else:
        print("  ❌ 정합성 실패 — 데이터 재검토 필요")
        return False

# ──────────────────────────────────────────────
# STEP 3: 데이터 토큰화
# ──────────────────────────────────────────────
def build_dataset(data_file, seq_len=SEQ_LEN):
    print("[STEP3] 데이터셋 토큰화 중...")
    tokenizer = KoJamoTokenizer()
    lines = open(data_file, encoding="utf-8").read().strip().splitlines()
    lines = [l for l in lines if l.strip()]

    all_tokens = []
    for line in lines:
        enc = tokenizer.encode(line + " ")  # (L, 6)
        all_tokens.append(enc)

    # 연결
    flat = torch.cat(all_tokens, dim=0)   # (N_total, 6)
    n = flat.shape[0]

    # 슬라이딩 윈도우 (stride = seq_len)
    windows_x, windows_y = [], []
    for i in range(0, n - seq_len - 1, seq_len):
        windows_x.append(flat[i   : i+seq_len])
        windows_y.append(flat[i+1 : i+seq_len+1])

    X = torch.stack(windows_x)  # (W, seq_len, 6)
    Y = torch.stack(windows_y)

    print(f"  ✅ 윈도우 {len(X)}개 생성 (총 {n:,} 토큰)")
    return X, Y, tokenizer

# ──────────────────────────────────────────────
# STEP 4: 5분 학습
# ──────────────────────────────────────────────
def train_5min(X, Y, tokenizer):
    print(f"[STEP4] 학습 시작 (최대 {TRAIN_LIMIT_SEC//60}분, device={DEVICE})")
    vocab_sizes = tokenizer.get_vocab_sizes()

    model = HRMContextNet(
        vocab_sizes=vocab_sizes,
        emb_dim=EMB,
        hidden_dim=HIDDEN,
        context_layers=CONTEXT_LAYERS,
        context_heads=CONTEXT_HEADS,
        max_seq_length=SEQ_LEN,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    dataset   = TensorDataset(X, Y)
    loader    = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True)

    best_loss = float("inf")
    start     = time.time()
    step      = 0
    epoch     = 0

    while True:
        epoch += 1
        total_loss = 0.0
        n_batches  = 0

        for xb, yb in loader:
            if time.time() - start >= TRAIN_LIMIT_SEC:
                break

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)  # 7-tuple of log_probs
            # 간단한 합산 loss
            loss = torch.zeros((), device=DEVICE)
            targets = [
                # type: sym>0→1, eng>0→2, num>0→3, else 0
                ((yb[:,:,3]>0).long() + 2*(yb[:,:,4]>0).long() + 3*(yb[:,:,5]>0).long()),
                yb[:,:,0], yb[:,:,1], yb[:,:,2],
                yb[:,:,3], yb[:,:,4], yb[:,:,5],
            ]
            for lgt, tgt in zip(logits, targets):
                loss = loss + nn.functional.nll_loss(
                    lgt.reshape(-1, lgt.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=0,
                )
            loss = loss / len(logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            step       += 1

            if step % 50 == 0:
                elapsed = time.time() - start
                print(f"  step={step:4d} | loss={total_loss/n_batches:.4f} | "
                      f"경과={elapsed:.0f}s / {TRAIN_LIMIT_SEC}s")

        if time.time() - start >= TRAIN_LIMIT_SEC:
            break

        avg = total_loss / max(n_batches, 1)
        print(f"  [epoch {epoch}] avg_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "vocab_sizes": vocab_sizes,
                        "emb_dim": EMB, "hidden_dim": HIDDEN,
                        "context_layers": CONTEXT_LAYERS, "max_seq_length": SEQ_LEN},
                       CKPT)

    elapsed = time.time() - start
    print(f"\n  ✅ 학습 완료: {step}스텝, {epoch}에폭, best_loss={best_loss:.4f}, 소요={elapsed:.0f}s")
    return model, tokenizer, best_loss

# ──────────────────────────────────────────────
# STEP 5: 생성 평가
# ──────────────────────────────────────────────
PROMPTS = [
    "대한민국의 수도는",
    "한국어는",
    "세종대왕은",
    "인터넷은",
    "태양은",
]

def evaluate(model, tokenizer, max_new=40):
    print("\n[STEP5] 생성 평가")
    model.eval()
    model.cpu()

    with torch.no_grad():
        for prompt in PROMPTS:
            enc = tokenizer.encode(prompt)  # (L, 6)
            ctx = enc.unsqueeze(0)           # (1, L, 6)

            # 간단한 greedy 생성
            # HRMContextNet forward를 step-by-step으로 호출
            generated = []
            for _ in range(max_new):
                inp = ctx[:, -SEQ_LEN:, :]
                logits = model(inp)
                # 마지막 위치
                type_lgt = logits[0][:, -1, :]
                pred_type = type_lgt.argmax(-1).item()

                token = [0, 0, 0, 0, 0, 0]
                if pred_type == 0:  # 한글
                    token[0] = logits[1][:, -1, :].argmax(-1).item()
                    token[1] = logits[2][:, -1, :].argmax(-1).item()
                    token[2] = logits[3][:, -1, :].argmax(-1).item()
                elif pred_type == 1:  # 기호
                    token[3] = logits[4][:, -1, :].argmax(-1).item()
                elif pred_type == 2:  # 영어
                    token[4] = logits[5][:, -1, :].argmax(-1).item()
                elif pred_type == 3:  # 숫자
                    token[5] = logits[6][:, -1, :].argmax(-1).item()

                new_tok = torch.tensor([[token]])  # (1, 1, 6)
                ctx = torch.cat([ctx, new_tok], dim=1)
                generated.append(token)

                # 개행이면 종료
                sym_id = token[3]
                if sym_id > 0:
                    char = tokenizer.reverse_sym.get(sym_id, "")
                    if char == "\n":
                        break

            gen_tensor = torch.tensor(generated)
            result = tokenizer.decode(gen_tensor)

            # 반복률 체크
            tokens_3gram = [result[i:i+3] for i in range(len(result)-2)]
            repeat = len(tokens_3gram) - len(set(tokens_3gram))
            repeat_rate = repeat / max(len(tokens_3gram), 1)

            print(f"\n  프롬프트: {prompt}")
            print(f"  생성:     {result[:80]}")
            print(f"  반복률:   {repeat_rate:.2f}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Ko-JamoNet 5분 Smoke Test")
    print("=" * 60)

    os.makedirs("train_data_hrm_pretrain_wiki", exist_ok=True)

    # 1. 위키 추출
    ok = extract_wiki(DATA_FILE, target=WIKI_SAMPLE)
    if not ok:
        sys.exit(1)

    # 2. 정합성 검사
    ok = validate_alignment(DATA_FILE)
    if not ok:
        print("  ⚠️ 정합성 실패지만 계속 진행 (경고 모드)")

    # 3. 데이터셋 구축
    X, Y, tokenizer = build_dataset(DATA_FILE)

    # 4. 학습
    model, tokenizer, best_loss = train_5min(X, Y, tokenizer)

    # 5. 평가
    evaluate(model, tokenizer)

    print("\n" + "=" * 60)
    print(f"✅ Smoke Test 완료! best_loss={best_loss:.4f}")
    if best_loss < 10.0:
        print("🟢 학습 수렴 신호 있음 → pretrain 계속 진행 가능")
    else:
        print("🔴 손실 정체 → 데이터/아키텍처 재검토 필요")
    print("=" * 60)
