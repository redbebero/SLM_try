# 모델/학습 공용 하이퍼파라미터 — train.py, train_sft.py, chat.py가 전부 여기서 가져다 씀.
# 값 하나 바꿀 때 여기만 고치면 됨 (예전에 HIDDEN_DIM이 파일마다 따로 있어서 어긋났던 버그 재발 방지).

EMB_DIM = 64
HIDDEN_DIM = 768
NUM_LAYERS = 8
DROPOUT = 0.1

# Stage 2: 실전 데이터(gsm8k+wiki 전체 문단, 문맥 1000자)로 확장
# Stage 1(짧은 단문, train_data_stage1)에서 epoch100까지 돌려 SOV 어순+실제 단어 형성 확인 후 전환.
# 되돌리려면: DATA_DIR="train_data_stage1", SEQ_LENGTH=64, STRIDE=16
DATA_DIR = "train_data_clean"
VAL_DATA_DIR = "val_data_clean"
SEQ_LENGTH = 250
STRIDE = 250
# attention은 O(seq_length^2)라 seq=1000에서 메모리가 급증함(seq=64 대비 약 250배) — 6GB 카드라 batch를 낮춤
# 실측(짧은 테스트): batch32=2.61GB, batch48=3.99GB — 그런데 실제 장시간 학습에서 batch48이
# 5.55GB까지 누적되어 OOM 발생(원인 미특정, torch.compile 껐음). 안전마진 크게 잡아 24로 낮춤.
# [업데이트] 커리큘럼 학습 가동: 길이를 250으로 대폭 축소하고 배치 크기를 96으로 4배 펌핑 (학습 초고속화)
BATCH_SIZE = 32

# Training stability / run control
# None = train indefinitely; Ctrl+C saves the current checkpoint safely.
EPOCHS = None
BASE_LR = 0.0005
WARMUP_STEPS = 300
CHECKPOINT_EVERY_EPOCH = 1
CHECKPOINT_PREFIX = "model_struct_v"
VAL_BATCHES = 50
# Match training inputs with inference inputs. Teacher forcing is reduced by stage.
TEACHER_FORCING_RATIOS = (1.0, 0.75, 0.5, 0.25, 0.0)
# Square-root inverse-frequency weights for jong classes in cleaned Hangul data.
JONG_CLASS_WEIGHTS = (
    0.209, 0.350, 1.252, 1.252, 0.209, 1.252, 1.252, 1.252,
    0.269, 1.252, 1.252, 1.252, 1.252, 1.252, 1.252, 1.252,
    0.519, 0.693, 1.252, 0.952, 0.477, 0.272, 1.252, 1.252,
    1.252, 1.252, 1.252, 1.252, 1.252,
)
# GRU + torch.compile is disabled by default while validating convergence.
USE_TORCH_COMPILE = False
