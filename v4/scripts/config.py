# 모델/학습 공용 하이퍼파라미터 — train.py, train_sft.py, chat.py가 전부 여기서 가져다 씀.
# 값 하나 바꿀 때 여기만 고치면 됨 (예전에 HIDDEN_DIM이 파일마다 따로 있어서 어긋났던 버그 재발 방지).

EMB_DIM = 64
HIDDEN_DIM = 768
NUM_LAYERS = 8
DROPOUT = 0.1

# 2026-07-18 재학습 실측: RTX 4050 6GB + torch 2.13/cu126 조합에서 hidden=768·8층 GRU가
# seq_length=64에서는 batch=32로 34에폭 문제없이 돌아가지만, seq_length=128부터는 배치를
# 8까지 낮춰도 첫 배치에서 즉시 OOM(항상 ~5.5GB 고정 소모 — cudnn 알고리즘 선택이 배치가
# 아니라 seq_length 문턱에서 급변하는 것으로 보임, 원인 미특정). SEQ_LENGTH=64/STRIDE=16을
# 이 하드웨어의 실측 안전값으로 고정. DATA_DIR은 그대로 실전 데이터(gsm8k+wiki 문단) 사용.
DATA_DIR = "train_data_clean"
VAL_DATA_DIR = "val_data_clean"
SEQ_LENGTH = 64
STRIDE = 16
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
# 28개 = tokenizer.get_vocab_sizes()의 jong 크기(0=빈종성 ~ 27=ㅎ)와 정확히 일치.
# 예전엔 29개짜리였음 — encode()가 절대 내지 않는(0-count) 29번째 클래스에 붙던
# 미관측 기본값(1.252, 다른 희귀 종성들과 동일)을 제거해 벡터 길이를 실제 vocab에 맞춤.
JONG_CLASS_WEIGHTS = (
    0.209, 0.350, 1.252, 1.252, 0.209, 1.252, 1.252, 1.252,
    0.269, 1.252, 1.252, 1.252, 1.252, 1.252, 1.252, 1.252,
    0.519, 0.693, 1.252, 0.952, 0.477, 0.272, 1.252, 1.252,
    1.252, 1.252, 1.252, 1.252,
)
# GRU + torch.compile is disabled by default while validating convergence.
USE_TORCH_COMPILE = False
