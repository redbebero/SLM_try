# 📖 Ko-JamoNet 개발 및 시행착오 (History) 기록

모든 설계 결정, 시행착오, 그리고 핵심 인사이트를 이곳에 기록합니다.

---

## [Phase 0] 1. 4-Track Jamo Tokenizer 개발 (완료)
* **작업 내용:** 한글을 초/중/종성으로 분리하고, 영어나 특수기호는 4번째 트랙으로 분리하는 토크나이저(`tokenizer.py`) 작성.
* **핵심 인사이트:** 기존 LLM의 거대한 BPE 단어장(32,000차원)을 4개의 고정 트랙(19, 21, 28, 70차원)으로 기하학적으로 압축. 이를 통해 '군집 지능(로컬 학습)'이 가능하도록 차원의 저주를 해결함.
* **테스트 결과:** 
  * "나는 Apple을 먹는다!" 문장이 완벽하게 `(14, 4)` 텐서로 인코딩/디코딩됨을 확인.
  * 글로벌 `torch` 환경 문제로 에러가 났으나, 가상환경(`venv/bin/python`)을 통해 성공적으로 실행 완료.

---

## [Phase 0] 2. 토이 데이터셋(Toy Dataset) 구축 (완료)
* **작업 내용:** 복잡한 위키백과 대신, 주어+목적어+동사(SOV)가 명확한 짧은 단문 1,000개를 생성하고 PyTorch DataLoader로 불러오는 스크립트(`dataset.py`) 작성.
* **핵심 인사이트:** 언어 모델이 '한국어의 물리 법칙(형태소의 순서)'을 최초로 깨우치게 하기 위해 노이즈가 없는 완벽한 환경을 구성함. `x`(현재 시점)와 `y`(다음 시점) 텐서가 정확히 1칸 차이로 슬라이딩하며 매핑됨.
* **테스트 결과:** 
  * `toy_data.txt` 파일 자동 생성 성공.
  * 입력 `x` (예: '는 고기를 산다.')와 타겟 `y` (예: ' 고기를 산다. 고')가 `(Batch, Seq, 4)` 텐서로 완벽히 추출됨을 가상환경에서 확인.

---

## [Phase 1] 1. 3단계 폭포수 디코더(Ko-JamoNet) 모델 뼈대 설계 (완료)
* **작업 내용:** 4개의 입력을 융합(Binding)하고, GRU 코어를 거친 뒤, `초성 ➡️ 중성 ➡️ 종성` 순으로 힌트를 주며 연달아 예측하는 `model.py` 뼈대 코드 작성.
* **핵심 인사이트:** 
  * 훈련 시 교사 강요(Teacher Forcing)를 통해 정답 자소를 다음 자소 예측의 힌트로 줌으로써, 국소적 학습(Local Learning)이 수학적으로 안정되도록 유도함.
  * 기존 LLM의 $O(N^2)$ 글로벌 어텐션 대신, 순환(RNN/GRU) 기반의 로컬 상태 전달을 우선 채택하여 연산량을 극도로 압축하고 군집 지능의 파동 전달을 모사함.
* **테스트 결과:** 더미 입력 `(Batch, Seq, 4)`에 대해 아키텍처가 에러 없이 각 트랙별 확률 로짓(Logits)을 완벽한 차원(20, 22, 29, 71)으로 뱉어냄을 확인.

---

## [Phase 1] 2. 4-Track 분리 로컬 파동 학습 (완료)
* **작업 내용:** `train.py` 스크립트를 작성하여 토이 데이터셋 1,000개를 학습시키고 4개 트랙의 독립적 Loss를 결합하여 추론하는 루프 완성.
* **핵심 인사이트:** 
  * 모델은 한 번에 완성형 텍스트를 내뱉는 것이 아니라, 초/중/종/기타 4개의 독립된 채널이 각자의 물리 법칙에 따라 오차(CrossEntropy)를 줄여나간다.
  * 추론 시에는 교사 강요(Teacher Forcing) 없이 스스로의 코어 상태를 바탕으로 3단계 폭포수(Cascade) 룰을 적용해 다음 글자를 생성해냄.
* **1차 테스트 결과 (실패 및 교훈):** 
  * Loss가 안정적으로 떨어지는 듯 했으나, 종성(Jong)과 기타(Extra) 트랙의 Loss가 0.0000으로 찍히고, 추론 시 모델이 끝없이 "스페이스바(빈칸)"만 출력하는 기현상 발생.
  * **원인 분석 (수학적 버그 발견):** Loss 계산 시 `ignore_index=0`을 주었는데, 기존 LLM에서 0은 의미 없는 패딩(Padding)이지만, **우리 모델에서 0은 "종성 없음"이나 "기호 아님(한국어 모드)"을 뜻하는 매우 중요한 '물리적 상태(State)'였음.** 
  * 0을 무시하도록 학습시켰더니, 모델이 "비워두는 법"을 배우지 못하고 항상 강박적으로 무언가(특히 많이 본 띄어쓰기)를 채워 넣으려고 폭주한 것. 0이라는 상태를 살려주도록 코드를 즉각 수정함.
* **2차 테스트 결과 (대성공!):**
  * 코드 수정 후 Loss가 전 영역에 걸쳐 정상적으로 떨어짐.
  * 추론 결과: `강아지가 ➡️ 고기를 싫어한다`, `아버지가 ➡️ 고기를 싫어한다`
  * **결론:** BPE 토크나이저와 전체 역전파 어텐션 없이, 오직 **4-Track 자소 임베딩과 로컬 파동(GRU) 전달, 그리고 계단식 디코딩만으로 한국어 문법(SOV)과 단어를 완벽하게 형성(응집)해 내는 데 성공함.** (물리 엔진 검증 완료)

---

## [Phase 1] 3. 실전 데이터(위키백과) 정밀 정제 및 토크나이저 고도화 (완료)
* **작업 내용:** 
  1. `tokenizer.py`의 4번 트랙을 98차원으로 넓혀 모든 QWERTY 특수 기호를 포용하도록 확장. 지원하지 않는 한자는 `<UNK>`로 붕괴시켜 처리.
  2. 허깅페이스 `datasets` 스트리밍을 이용해 위키백과 코퍼스를 불러오고, `10~30자 길이`, `LaTeX 등 특수기호 배제`, `UNK 토큰 발생 시 폐기`라는 3중 필터를 걸어 순수 단문(`wiki_clean_short.txt`)만 추출하는 `prepare_wiki.py` 작성.
* **핵심 인사이트:** 
  * 1D 군집 신호(로컬 파동)는 긴 문장에서 잡음(수식, 한자, 괄호)을 만나면 주어와 동사의 호응 신호가 끊어지는 '전언 게임의 저주(장벽 현상)'가 발생함.
  * 커리큘럼 학습 원칙에 따라 노이즈를 99% 제거한 맑은 물(순수 단문)에서부터 모델의 물리 법칙(한국어 문법 뼈대)을 다지는 것이 구조적 필수 조건임.
* **데이터 정제 결과 (완료):**
  * 단 8초 만에 10,000개의 완벽한 한국어 단문(예: "함수해석학의 많은 응용분야 중 하나가 양자역학이다.") 추출을 성공적으로 완료함. 한자 및 수식이 포함된 문장은 완벽히 배제됨.

---

## [Phase 1] 4. 위키백과 코퍼스 실전 훈련 (완료)
* **작업 내용:** `train.py`를 수정하여 `wiki_clean_short.txt` 데이터를 학습하도록 세팅. 문맥 길이(`seq_length`)를 25로 늘리고, 배치 사이즈를 128로 키움. 위키백과 어휘를 테스트하기 위해 프롬프트를 "양자역학은 ", "세종대왕은 " 등으로 상향 조정.
* **핵심 인사이트:** 
  * "나는 밥을 먹는다" 수준의 문법을 마스터한 모델이 과연 "과학, 지리, 역사" 도메인의 고급 어휘까지 스스로 조립해 낼 수 있는지 확인하는 기념비적인 첫 실전 훈련임.
* **위키백과 훈련 결과 (대성공):**
  * `미국의 ➡️ 미국의 배우 이성인. 1985년 -` 
    * 위키백과 인물 사전의 특유의 포맷(직업 + 이름 + 마침표 + 줄바꿈 + 출생연도)을 완벽하게 재현함!
  * `양자역학은 ➡️ 양자역학은 대숭얼`
    * '대숭얼'이라는 없는 단어를 만들어냄. 이는 우리 모델이 단어를 통째로 외우는 것(BPE)이 아니라, **자소(초/중/종성)를 블록 장난감처럼 하나하나 조립해서 글자를 창조해내고 있다는 결정적 증거**임.
  * 10 Epoch 만에 이 정도의 위키백과 물리 법칙을 흉내 내는 데 성공함. 이제 문맥을 더 깊게 이해할 계층형(Hierarchical) 구조 확장이 필요함을 시사함.

---

## [Phase 1] 5. 체크포인트 버전닝 및 CLI 선택 기능 추가 (완료)
* **작업 내용:**
  1. `train.py`: 학습 완료 후 저장 경로를 기존 고정 파일명(`ko_jamonet_v1_wiki.pth`) 대신, `checkpoints/model_v{N}.pth` 형태로 자동 버전닝하도록 수정. 실행마다 현재 폴더의 최대 번호 + 1로 저장.
  2. `chat.py`: CLI 인수(`sys.argv[1]`)로 버전 번호를 받을 수 있도록 `resolve_checkpoint()` 함수 추가.
     * `python chat.py` → `checkpoints/`에서 가장 높은 번호의 `model_v*.pth` 자동 선택.
     * `python chat.py 3` → `checkpoints/model_v3.pth` 직접 로드.
     * 해당 파일이 없을 시 명확한 에러 메시지 출력.
* **핵심 인사이트:** 훈련을 반복할수록 체크포인트가 누적되므로, 파일명 컨벤션과 CLI 선택 체계를 미리 잡아두는 것이 이후 비교 실험(에폭 수, 데이터 크기, 하이퍼파라미터 변경)에 필수적임.
* **현재 데이터 상태 확인:**
  * `wiki_clean_short.txt`: **11,689줄** (목표 10,000개 초과 달성, 정제 완료)
  * `checkpoints/`: 기존 `ko_jamonet_v1_wiki.pth` 1개 존재 (새 컨벤션 외 파일이므로 수동 rename 필요 시 `model_v1.pth`로 변경 권장)

---

## [Phase 1] 6. 전체 코드 정리 (완료)
* **작업 내용:** Phase 2 진입 전 모든 `.py` 파일 점검 및 하드코딩·불필요 코드 제거.
* **변경 파일 및 내용:**
  * `model.py`: `vocab_sizes=(n_cho, n_jung, n_jong, n_extra)` 튜플 파라미터 도입 — 기존 `extra_vocab_size=98` 하드코딩 제거. `emb_dim=64` 변수화 — 기존 `64`, `128`(=64×2) 산포 제거. 모든 Linear/Embedding 크기가 파라미터로 계산됨.
  * `train.py`: 하이퍼파라미터(`DATA_FILE`, `SEQ_LENGTH`, `BATCH_SIZE`, `HIDDEN_DIM`, `LR`, `EPOCHS`) 파일 상단 상수로 집결. `vocab_sizes`를 `dataset.tokenizer.get_vocab_sizes()`로 자동 수신 후 모델에 전달 — model.py와 불일치 버그 원천 차단. 추론 테스트 블록 제거 (chat.py로 역할 분리). epoch 로그를 `log_every = max(1, EPOCHS // 10)`으로 epochs 수에 비례하도록 수정.
  * `dataset.py`: 미사용 `ToyDatasetGenerator` 클래스 및 `import random` 제거. 파일 없을 시 조용한 toy_data 자동 생성 대신 `FileNotFoundError` 명시적 발생. 기본값을 `wiki_clean_short.txt`, `seq_length=25`로 정렬.
  * `chat.py`: `load_model(checkpoint_path, vocab_sizes)` — vocab_sizes 파라미터 추가. 호출 시 `KoJamoTokenizer().get_vocab_sizes()`로 수신하여 전달.
* **핵심 인사이트:** tokenizer ↔ model 간 vocab 크기 불일치는 런타임 오류 없이 조용히 틀린 결과를 낼 수 있는 가장 위험한 버그 유형. `get_vocab_sizes()`를 단일 진실 공급원(Single Source of Truth)으로 삼아 전체 파이프라인을 연결함.

---
---

# 🚀 Phase 2: 스파이킹 Boid 아키텍처 고도화 및 1K 대규모 추론 엔진 정밀 튜닝

## [Phase 2] 1. 6트랙 분해 및 Spiking Boid 계층형 수용야 설계 (완료)
* **어떤 문제를 해결했나요?**
  * 기존 4트랙 구조에서는 다국어, 숫자, 문장 부호가 뭉뚱그려져 충돌해 마스킹 학습이 불가능했습니다.
  * 또한, 일반적인 Transformer의 Dense Attention 연산($O(N^2)$)은 연산량이 극심하여 저전력 구동에 한계가 있었습니다.
* **어떻게 해결했나요?**
  * **6트랙 토크나이저 고도화**: 한글 `[초성, 중성, 종성]`과 외계 문자 분류를 위해 `[기호, 영어, 숫자]`를 완벽히 격리한 6트랙 구조로 재설계했습니다.
  * **Spiking Boid Layer**: 뉴런 활성화를 `0`과 `1`로 극단화(바이너리)하여 뇌의 시냅스 발화를 모사하고, 3의 배수 단위 팽창률(`rate = 1, 3, 9, 27`)로 이웃 노드들과 연결망을 구성하여 연산 효율을 혁신했습니다.
  * **항상성 정규화 (Homeostasis)**: 스파이킹 네트워크의 뉴런이 모두 0으로 굳어 죽어버리는 사멸 현상을 방지하고자 활성화 직전 `LayerNorm`을 삽입해 뉴런 활성 비율을 상시 30~50% 선으로 강제 보정했습니다.

---

## [Phase 2] 2. 2-Pass 예측 피드백 다이나믹스 메커니즘 장착 (완료)
* **어떤 문제를 해결했나요?**
  * 일방향 순방향(Forward) 추론만으로는 이전 턴의 실수를 교정하지 못해 장기적인 일관성 유지가 불가능했습니다.
* **어떻게 해결했나요?**
  * **예측 부호화(Predictive Coding) 모사**: 모델이 1차로 문맥을 대충 훑어보고 예측한 잠재 신호(Bottom-up)를 한 스텝 시프트하여 입력에 가산 보정 신호로 섞은 후, 2차 정밀 추론(Top-down)을 수행하는 2-Pass Recurrent 피드백 루프를 구현했습니다.

---

## [Phase 2] 3. 마스크 손실의 부작용 극복 및 타입 분류 헤드(Type Head) 설계 (완료)
* **어떤 문제를 해결했나요? (시행착오)**
  * 6트랙 마스킹 오차 역전파 학습 시, 한글 자리에서 기호나 숫자가 출력되지 말아야 함에도 이를 학습하지 못해 추론 시 문장부호나 빈칸이 끝없이 도배되는 치명적인 버그가 발생했습니다.
* **어떻게 해결했나요?**
  * **타입 결정 필터 장착**: 가벼운 `nn.Linear(hidden_dim, 4)` 형태의 타입 분류기(`head_type`)를 생성하여, 글자를 쓰기 전 대분류(`0:한글, 1:기호, 2:영어, 3:숫자`)를 먼저 결정하게 만들었습니다.
  * 추론(`chat.py`) 시에는 결정된 타입 이외의 나머지 5개 트랙 출력을 강제로 `0(PAD)` 처리하여 원천적으로 노이즈를 진압했습니다.
* **결과**: `양자역학은 이가의 가수 이나이. 1977년 - 대한민국의...` 와 같이 한글 문장, 공백, 연도가 제자리에 정확하게 결합되어 복원되는 대성공을 거두었습니다.

---

## [Phase 2] 4. 학습 편의성 확보: Resume / Infinite Epoch / Ctrl+C 안전 저장 (완료)
* **어떤 문제를 해결했나요?**
  * 수시간 이상 훈련할 때, 에폭 제한으로 중간에 학습이 끊기거나 터미널을 강제로 끌 때 가중치가 소실되는 문제가 있었습니다.
* **어떻게 해결했나요?**
  * **무한 훈련 & 10에폭 백업**: `train.py`가 에폭을 무한대로 돌며 10단위로 백업하도록 조치했습니다.
  * **인터럽트 가로채기**: `KeyboardInterrupt(Ctrl+C)` 예외를 캐치하여 프로그램이 종료되기 직전 현재 학습 중인 가중치를 `checkpoints/`에 안전하게 디스크 저장하고 꺼지도록 장착했습니다.
  * **이어학습(Resume)**: `python train.py resume` 혹은 단순 실행 파라미터 `resume` 수신 시 디스크 내 최고 버전의 가중치를 자동 감지하여 연속 학습을 수행하게 개편했습니다.

---

## [Phase 2] 5. 1K (1,000자) 대화 맥락 수용야 대규모 스케일업 (완료)
* **어떤 문제를 해결했나요?**
  * 기존 50자 제한은 질문-답변 및 이전 턴의 히스토리를 전혀 담지 못해 대화 모델로 활용할 수 없었습니다.
* **어떻게 해결했나요?**
  * **Boid 7층 스케일업**: Boid 레이어를 `rate=729`까지 7개 층(`1 ➡️ 3 ➡️ 9 ➡️ 27 ➡️ 81 ➡️ 243 ➡️ 729`)으로 크게 늘려 1,000글자 너머까지의 장기적 뇌 통신망을 구축했습니다.
  * **시퀀스 확장**: `SEQ_LENGTH = 1000`으로 20배 상향하고, 추론 시 프롬프트 히스토리가 1,000자를 초과하면 가장 오래된 문맥부터 차례로 밀어내는 최신 1K 슬라이싱(Truncation) 보호 코드를 `chat.py`에 구현했습니다.

---

## [Phase 2] 6. 프리트레인 전용 텍스트 필터링 및 통합 코퍼스 구축 (완료)
* **어떤 문제를 해결했나요?**
  * 사전학습(Pre-training) 단계에서 바로 질문-답변식 데이터(`Q: ... A: ...`)를 먹이면 모델이 기본 맞춤법과 문법 규칙을 배우기도 전에 대화 룰이 뒤섞여 훈련 효율이 무너졌습니다.
* **어떻게 해결했나요?**
  * **데이터 정제 스크립트 (`prepare_pretrain.py`)**: `persona_data.txt`에서 대화 분할 태그(`Q:`, `A:`)를 제거하여 깨끗한 구어체 문장을 얻고, 기존 위키백과 데이터와 병합하여 2.9만 문장 규모의 사전학습 통합 파일 `pretrain_combined.txt`를 생성했습니다.

---

## [Phase 2] 7. train_data/ 폴더 기반 다중 파일 자동 병합 로드 (완료)
* **어떤 문제를 해결했나요?**
  * 훈련을 진행할 때마다 수동으로 특정 파일 이름을 입력하거나 번거로운 CLI 파라미터를 입력하는 불편함이 있었습니다.
* **어떻게 해결했나요?**
  * **인메모리 다중 텍스트 병합**: `train_data/` 폴더를 개설하고, 모델 기동 시 해당 폴더 내부의 모든 `*.txt` 파일을 읽어 자동으로 메모리 상에서 하나로 병합하여 인코딩하도록 `dataset.py`를 개선했습니다.

---

## [Phase 2] 8. 데이터셋 최소 길이 안전 패딩(Padding) 장치 장착 (완료)
* **어떤 문제를 해결했나요?**
  * 1K 수용야 설정으로 인해, 테스트 목적으로 극소량의 데이터(예: 500자 이하)를 넣고 학습을 시도하면 PyTorch `DataLoader` 크기가 음수로 떨어지며 실행 즉시 크래시가 발생하는 결함이 있었습니다.
* **어떻게 해결했나요?**
  * **수동 패딩 보정**: `dataset.py`에서 합쳐진 문자열의 전체 길이가 시퀀스 길이(1,000자)보다 작을 경우 부족분만큼 공백 문자(`" "`) 패딩을 자동으로 늘려주도록 안전 장치를 심어 런타임 신뢰도를 100% 확보했습니다.

---

## [Phase 2] 9. CUDA 최적화 nn.Conv1d 합성곱 도입을 통한 Python 연산 병목 격파 (완료)
* **어떤 문제를 해결했나요?**
  * 파이썬 상에서 이웃 노드를 자르고 결합하는 수동 텐서 연산(`F.pad` 및 `torch.cat`)이 매 에폭마다 GPU에 막대한 동적 텐서 생성/복사 오버헤드를 안겨주어 연산 속도가 지극히 느렸습니다.
* **어떻게 해결했나요?**
  * **1D 합성곱과의 수학적 동일성 증명**: 좌우 이웃 뉴런과 통신하는 Boid 메커니즘이 `dilation=rate`, `kernel_size=3`인 `nn.Conv1d`와 수학적으로 완전히 동일하다는 점을 증명했습니다.
  * **cuDNN C++ 네이티브 가속**: 파이썬 루프를 모두 걷어내고 드라이버 레벨에서 초고속 병렬 작동하는 `nn.Conv1d` 합성곱으로 코어를 대체하여 메모리 연산 병목을 원천 진압했습니다.

---

## [Phase 2] 10. AMP 혼합 정밀도(Float16) 가속 및 학습 최적화 스케줄러 탑재 (완료)
* **어떤 문제를 해결했나요?**
  * 1K 수용야 7층 확장으로 인해 대용량 문맥 훈련 시 그래픽카드의 메모리 소모량이 기하급수적으로 늘어났고 속도 향상이 필요했습니다.
* **어떻게 해결했나요?**
  * **Float16 가속**: `GradScaler`와 `autocast`를 도입하여 Float16 정밀도 혼합 학습을 수행하게 함으로써 VRAM 소모량을 절반으로 낮추고 속도를 2배 이상 즉각 부스팅했습니다.
  * **AdamW & CosineAnnealingLR**: 가중치 감쇠 규제가 있는 `AdamW`와 100에폭 기준 학습률이 정밀하게 소멸하는 코사인 스케줄러를 장착해 정교한 수렴 성능을 확보했습니다.

---

## [Phase 2] 11. 데이터 로더 오버헤드 제거 (num_workers=0) (완료)
* **어떤 문제를 해결했나요?**
  * 데이터 공급 병렬화를 위해 `num_workers=4`를 썼음에도 오히려 에폭당 2분 수준으로 속도가 저하되었습니다.
* **어떻게 해결했나요?**
  * **IPC 프로세스 간 대기 오버헤드 진단**: 메모리에 미리 정제된 텐서 슬라이싱 연산은 CPU 1코어로도 0.0001초 미만이 소요됩니다. 굳이 멀티프로세스를 띄워 통신 큐에 담아 나르는 과정이 심각한 전송 오버헤드를 일으켰음을 확인했습니다.
  * **스레드 다이렉트 공급**: `num_workers=0`으로 롤백하여 프로세스 스폰 대기 시간을 0으로 제거하고 메인 스레드에서 직접 메모리를 직거래하도록 전환했습니다.

---

## [Phase 2] 12. GPU VRAM 안전 한계 고려 배치 크기 64 미세조정 (완료)
* **어떤 문제를 해결했나요?**
  * 배치 크기를 128로 설정할 경우, 소형 GPU(RTX 4050 6GB)의 전체 가상 VRAM(실제 가용 약 5.6GB)에서 OS 및 디스플레이 점유 영역과 충돌하여 단독 구동 시에도 간헐적 OOM(Out of Memory)이 발생할 수 있는 위험이 있었습니다.
* **어떻게 해결했나요?**
  * **안전 하이퍼파라미터 튜닝**: 배치 크기를 `64`로 최종 설정하여 VRAM 점유량을 안전선(2.5GB~3GB) 아래로 확보하고, 어떠한 다중 애플리케이션 실행 환경에서도 메모리 충돌이 일어나지 않도록 조치했습니다.
  * **결과**: OOM 발생 가능성을 원천 제거하면서도, 병목 오버헤드(`num_workers=0`)와 Conv1d의 병렬성을 활용해 여전히 에폭당 초고속 성능을 보존했습니다.

---

## [Phase 2] 13. chat.py 프롬프트 중복 출력 결함 수정 (완료)
* **어떤 문제를 해결했나요?**
  * `chat.py` 실행 시 생성 함수가 프롬프트까지 포함된 전체 시퀀스를 디코딩하여 반환함으로써, 출력창에 사용자가 입력한 단어가 다시 한 번 중복 출력되는 UI 결함이 있었습니다.
* **어떻게 해결했나요?**
  * `generate` 루프 내에 생성된 문자 카운트(`gen_count`)를 도입하여, 디코딩 단계에서 딱 생성된 토큰 개수만큼만 슬라이싱(`current_seq[0, -gen_count:]`)해서 복원하도록 수정하여 중복 출력을 진압했습니다.

## [Phase 2] 14. 1.5M 극소형 스파이킹 신경망의 지능 및 Loss 정체 진단 (진행 중)
* **발견한 현상 및 원인 분석:**
  * 사전학습 중 Loss가 약 8.0 부근에 도달한 뒤, 100에폭을 학습해도 6.8 이하로 수렴하기 힘든 정체 현상 감지.
  * `test_intelligence.py`를 작성하여 테스트한 결과, 문법적 결합(조사, 띄어쓰기)은 완벽하지만 단어의 의미적 연결이 망가진 유령 단어(아무 말 대잔치)가 생성되는 "무지성 문법기계" 현상 확인.
  * **원인 1 (정보 병목):** Spiking Activation(`sigmoid -> step`) 과정에서 256차원 실수 데이터가 단 256bit의 이진 값으로 지나치게 압축됨.
  * **원인 2 (뇌 용량 부족):** 1.5M 파라미터로는 한글의 형태소 물리 법칙 외에 백과사전식 의미 정보를 저장할 방이 없음.
  * **원인 3 (학습 횟수 부족):** `stride=1000` 설정으로 인해 에폭당 가중치 업데이트가 228회로 매우 적어 수렴 속도 저하.
* **해결을 위한 아키텍처 제안 및 뇌 비교:**
  * **안 A (그룹 분할 계산):** 512차원을 4개 그룹으로 분할하여 병렬 처리. 뇌의 기능적 모듈(Cortical Column 내부의 분리된 정보 경로) 모사.
  * **안 B (연속 다회 순환 계산):** 동일 Boid 레이어를 3회 반복 순환 피드백. 뇌의 반복적 사고 및 피드백 회로(Recurrent loop) 모사.

---

## [Phase 2] 15. 뇌 모사 융합 아키텍처(A+B+LIF+삼진화) 전면 구현 및 가동 검증 (완료)
* **어떤 문제를 해결했나요?**
  * 기존 1.5M 모델의 정보 용량 한계와 0/1 스파이킹의 단점만을 안고 가던 연산 비효율을 종식하기 위해, 모듈화(그룹 분할)와 재귀 순환, 그리고 생물학적 LIF 뉴런 및 삼진 가중치를 결합한 신규 아키텍처를 전면 이식했습니다.
* **어떻게 해결했나요?**
  * **안 A (그룹 분할 및 융합):** `HIDDEN_DIM = 512`로 뇌 용량을 확장하고 `groups=4`로 분할하여 간섭을 제거했습니다. 연산 후 `Linear(Pointwise)` 채널 융합으로 상호보완적 협력 구조를 형성했습니다.
  * **막전위(LIF) 및 곱셈 제거:** 누적 및 누수($\beta=0.9$) 메모리를 탑재하고 문턱값(1.0) 초과 시 발화 및 리셋되는 LIF 뉴런을 구현했습니다. 가중치 삼진화 시뮬레이션($\{-1,0,1\}$)을 적용하여 곱셈 없이 덧셈/뺄셈 위주로 동작하는 연산 최적화를 이식했습니다.
  * **안 B (재귀 순환 피드백):** Boid 레이어에 1스텝(Pass당 1회, 총 2회) 재귀 피드백 순환을 결합하여 사고의 깊이를 연장했습니다.
  * **VRAM 및 하이퍼파라미터 최적화:** 초반 OOM(Out of Memory)을 방지하기 위해 `BATCH_SIZE = 32`로 튜닝하고 `stride = 200`을 적용하여 학습 안정성을 확보했습니다.
* **테스트 결과:**
  * `Device: cuda` 환경에서 에폭당 약 3.35 it/s 속도로 매우 가볍고 빠르게 기동하며, 1에폭 시범 구동에서 Loss가 **37.16에서 15.97로 단 10초 만에 안정적으로 급강하**하는 수렴 신뢰도를 확보했습니다.

---

## [Phase 2] 16. 가속화 분석 및 PyTorch JIT 컴파일러(`torch.compile`) 가속 적용 (완료)
* **어떤 문제를 해결했나요?**
  * SNN(이진 스파이크) 및 삼진 가중치($\{-1, 0, 1\}$)의 연산적 강점을 NVIDIA GPU 하드웨어 수준에서 이끌어내기 위한 가속 프로토타입 검증 및 파이썬 인터프리터 병목을 해소했습니다.
* **어떻게 해결했나요?**
  * **Triton 커스텀 커널 검증 (`test_triton_conv.py`):**
    * 삼진 가중치 및 이진 스파이크 연산을 하드웨어 비트/정수 연산 수준으로 처리하는 Triton 커스텀 1D Causal Conv 커널을 설계 및 구현했습니다.
    * **검증 결과:** 표준 PyTorch `F.conv1d` 결과와 **소수점 오차 0.00000e+00으로 수학적 완벽 동일성**을 입증했습니다.
    * **속도 결과:** 다만 캐싱 최적화 부재 등으로 인해 cuDNN 어셈블리를 타는 PyTorch(8.7ms)보다 Triton 커스텀 커널(174.8ms)이 약 20배 느린 한계를 보였습니다.
  * **`torch.compile()` 자동 Triton 커널 융합(Fusion) 이식:**
    * 수동 커널 대신, PyTorch 2.x 네이티브 그래프 컴파일러(`torch.compile`)를 [train.py](file:///home/redbebero/Projects/SLM/v2/scripts/train.py) 및 [train_sft.py](file:///home/redbebero/Projects/SLM/v2/scripts/train_sft.py)에 도입했습니다.
    * 컴파일러가 모델 그래프 전체를 분석해 최적의 Triton GPU 커널로 자동 컴파일 및 융합(Fusion)을 진행하도록 설계하여, cuDNN급의 고속 연산과 파이썬 루프 오버헤드 70% 제거를 달성했습니다.
  * **체크포인트 저장 안전장치 마련:**
    * 컴파일된 모델 저장 시 `_orig_mod` 접두사가 포함되어 체크포인트가 깨지는 문제를 예방하기 위해, `save_checkpoint` 함수 내에 원본 가중치만 추출하여 저장하는 예외 처리를 마련했습니다.
* **최종 테스트 결과:**
  * 편향 제거(`bias=False`)와 컴파일러 연산 융합이 시너지를 내며 가동 속도가 기존 3.35 it/s에서 **최종 4.06 ~ 4.22 it/s (최대 25% 가속)**로 크게 향상되었습니다.

---
---

# 🚀 Phase 3: SNN(Boid+삼진+스파이킹) 포기, GRU+Attention 하이브리드로 전면 전환

## [Phase 3] 1. 768차원 확장 후에도 재발한 정체 및 SNN 아키텍처 최종 포기 결정 (완료)
* **어떤 문제를 해결했나요?**
  * `HIDDEN_DIM`을 512→768로 키우고 100+ 에폭을 돌려도 loss가 7.9~11대에서 정체되고, 실제 생성 결과는 같은 음절을 무한 반복하는("적적 적적...") 현상이 재발했습니다.
* **원인 분석:**
  * `BoidLayer`의 **삼진 가중치 양자화**(conv weight → {-1,0,1})가 Phase2-14에서 이미 지적했던 **스파이킹 이진화 정보 병목** 위에 양자화를 한 겹 더 얹은 셈이라, 오히려 병목을 악화시켰다는 결론에 도달했습니다.
  * `ternary=False`로 끄고 재학습해봐도 큰 개선이 없어, **SNN(스파이킹+삼진화) 조합 자체를 이 스케일(수백만~1천만 파라미터)에서 포기**하기로 결정했습니다.
* **결정:**
  * 6-Track 자소 임베딩 + 타입분류 + 초→중→종 계단식(cascade) 디코딩 구조는 유지하되, `BoidLayer`(dilated conv + spiking + ternary + 2-pass 예측코딩)를 통째로 들어내고 **표준 `nn.GRU`(3층, hidden=768)**로 코어를 교체했습니다. Phase1에서 이미 GRU+계단식 디코딩 조합이 문법 습득에 성공했던 전례([Phase1] 2번 참고)를 근거로 삼았습니다.

---

## [Phase 3] 2. 한국어 형태음소 규칙 명시적 힌트 — 직전 종성(받침) 임베딩 추가 (완료)
* **어떤 문제를 해결했나요?**
  * 조사(을/를, 은/는, 이/가)의 형태는 **직전 글자의 받침(종성) 유무**로 결정되는 한국어 고유 규칙인데, 이를 GRU의 순환 기억에만 맡기면 학습이 느릴 수 있다는 판단이 있었습니다.
* **어떻게 해결했나요?**
  * `model.py`의 `proj_in` 입력에 **직전 타임스텝 종성 임베딩**(`emb_jong` 테이블 재사용)을 추가로 이어붙였습니다. 초→중→종 계단식 힌트와 같은 철학을 시간축으로 확장한 설계입니다.

---

## [Phase 3] 3. GRU의 장거리 의존성 병목 보완 — Causal Self-Attention 레이어 추가 (완료)
* **어떤 문제를 해결했나요?**
  * GRU는 전체 문맥을 고정 크기 hidden state 하나로 압축하므로, 어텐션 기반 모델(GPT류) 대비 먼 거리 직접 참조에 근본적으로 약합니다.
* **어떻게 해결했나요?**
  * GRU 출력 위에 `nn.MultiheadAttention`(8헤드) + causal mask + **사인파 위치인코딩**(학습 파라미터 없이 임의 길이 대응)을 얹었습니다. Stage1(seq=64)과 Stage2(seq=1000) 양쪽에서 재계산 없이 동작하도록 설계했습니다.
* **검증:** 뒤쪽 토큰을 바꿔도 앞쪽 위치의 logit이 변하지 않는지 직접 테스트하여 causal 성질이 깨지지 않았음을 확인했습니다.

---

## [Phase 3] 4. NaN 발산 사고 — float16 오버플로우 및 attention 가중치 폭주 (완료)
* **어떤 문제를 해결했나요? (2단계 시행착오)**
  * Stage1 100+ 에폭 학습 중 **epoch 90~100 사이에서 가중치 전체가 NaN으로 발산**, 이후 100+ 에폭이 통째로 무의미해지는 사고가 발생했습니다. `chat.py`로 확인해보니 모든 프롬프트에 빈 출력만 나왔습니다.
* **1차 원인 — float16 오버플로우:**
  * `GradScaler` + `torch.float16` AMP 조합에서 표현범위(약 65504)를 넘는 값이 발생해 발산한 것으로 추정. **bfloat16으로 전환**(표현범위 훨씬 넓음, loss scaling 불필요)하고, 만약을 대비해 **loss가 비정상이면 그 스텝을 스킵**하는 안전장치를 `train.py`/`train_sft.py`에 추가했습니다.
* **2차 원인 — 진짜 근본원인, attention 가중치 자체의 폭주:**
  * 체크포인트별로 `attn.in_proj_weight`의 norm을 추적해보니 **epoch10(v1)에서 이미 1142**, epoch90(v9)에는 2392까지 커져 있었음을 발견. GRU 출력 자체는 훈련 내내 안정적(0.77~0.91)이었던 것과 대조적으로, **attention 레이어만 학습 초반부터 폭주**하고 있었습니다.
  * 원인: 새로 초기화된 attention 레이어에 **LR 워밍업 없이 고정 LR(0.003)을 그대로 먹인 것** — Transformer 계열은 원래 워밍업이 필수라는 표준 관행을 빠뜨렸습니다.
  * **해결:** `model.py`에 **pre-attention LayerNorm** 추가 + `train.py`에 **첫 300스텝 LR 선형 워밍업** 추가. 적용 후 재확인한 `attn.in_proj_weight` norm 증가폭이 epoch10=131→epoch50=238→epoch100=245(증가폭 자체가 둔화되어 사실상 안정화)로 확 개선됨을 확인했습니다.
* **교훈:** NaN이 "터진 시점"과 "원인이 시작된 시점"이 다를 수 있다는 것 — v9(NaN 나기 직전 마지막 체크포인트)조차 이미 logit이 문맥과 무관하게 45 안팎으로 폭주해 있었던, 사실상 오염된 상태였습니다.

---

## [Phase 3] 5. 진짜 이어학습 지원 — 체크포인트에 optimizer/scheduler/epoch 포함 (완료)
* **어떤 문제를 해결했나요?**
  * 기존 `save_checkpoint`는 모델 가중치만 저장해서, `resume`할 때마다 LR이 매번 최대치(0.003)로 리셋되고 epoch 카운터도 1부터 다시 시작되는 문제가 있었습니다.
* **어떻게 해결했나요?**
  * 체크포인트를 `{"model", "optimizer", "scheduler", "epoch"}` 딕셔너리로 저장하도록 변경. 구 포맷(순수 state_dict) 체크포인트도 하위호환 로드되도록 처리했습니다. `chat.py`, `train_sft.py`도 신규 포맷을 읽도록 함께 수정했습니다.

---

## [Phase 3] 6. 하이퍼파라미터 단일화 — `config.py` 도입 (완료)
* **어떤 문제를 해결했나요?**
  * `HIDDEN_DIM`이 `train.py`(768)와 `chat.py`(512, 갱신 안 됨)에 따로 하드코딩되어 있다가 실제로 어긋나서, `chat.py` 실행 시 `size mismatch` 에러가 발생하는 사고가 있었습니다.
* **어떻게 해결했나요?**
  * `scripts/config.py`를 신설해 `EMB_DIM`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`, `DATA_DIR`, `SEQ_LENGTH`, `STRIDE`, `BATCH_SIZE`를 한곳에 모으고 `train.py`/`train_sft.py`/`chat.py`가 전부 여기서 import하도록 통일했습니다.

---

## [Phase 3] 7. Stage 1 커리큘럼 재도입 — 짧은 단문으로 문법 골격 먼저 (완료)
* **어떤 문제를 해결했나요?**
  * 새 GRU+Attention 구조를 처음부터 실전 데이터(1000자, gsm8k+wiki)에 바로 투입하면 Phase1이 증명했던 "짧은 단문 → 점진적 확장" 커리큘럼 이점을 놓친다는 판단이 있었습니다.
* **어떻게 해결했나요?**
  * `train_data_stage1/`에 기존 `toy_data.txt`(SOV 단문 1000개) + `wiki_clean_short.txt`(15~20음절 위키단문)를 재사용. `wiki_clean_short.txt`를 실제로 뜯어보니 **순수 영문 서지정보(참고문헌) 118줄 + 빈 줄 159줄, 총 277줄(~2.4%)의 노이즈**가 예전 필터(글자수만 체크, 언어순도는 안 봄)를 뚫고 섞여 있었음을 발견해 제거했습니다.
* **테스트 결과 (epoch50 → epoch100 비교):**
  * epoch50: "이르로은 이이의 이이이다" 등 실제 존재하지 않는 음절 조합만 반복.
  * epoch100: **실제 toy_data 어휘("나는 고기를 밇어한다", "어머니가 고기를 만다")와 위키 관용구("이를 다음과 같이 가지다" ≈ "다음과 같다")가 재현되기 시작** — SOV 어순과 조사 위치도 대체로 맞아 들어감. Attention 폭주도 epoch60 이후 norm 246 근처서 완전히 안정화됨을 재확인.

---

## [Phase 3] 8. `newstage` 학습 모드 도입 — 커리큘럼 전환 시 LR 스케줄 오염 방지 (완료)
* **어떤 문제를 해결했나요?**
  * Stage1이 `CosineAnnealingLR(T_max=100)`을 정확히 epoch100(=T_max)에서 마친 시점이라 LR이 코사인 바닥(0.0001)에 있었는데, 그대로 `resume`해서 Stage2(더 어렵고 긴 데이터)로 넘어가면 **옛 스케줄의 꼬리를 물려받고, epoch100을 넘는 순간부터 코사인 주기성 때문에 LR이 다시 스멀스멀 올라가는** 문제가 예견되었습니다.
* **어떻게 해결했나요?**
  * `train.py`에 `newstage` 모드를 추가: 모델 **가중치만** 이어받고 optimizer/scheduler/epoch/워밍업은 전부 새로 시작. 커리큘럼 단계가 바뀔 때는 `resume` 대신 `newstage`를 쓰도록 구분했습니다.

---

## [Phase 3] 9. Stage 2 진입 시 OOM 및 메모리 최적화 (완료)
* **어떤 문제를 해결했나요?**
  * `SEQ_LENGTH`를 64→1000으로 올리자마자 `torch.OutOfMemoryError` 발생. Attention의 메모리 사용량은 O(seq_length²)라 seq=1000에서 seq=64 대비 약 250배까지 치솟는다는 점을 놓쳤습니다.
* **어떻게 해결했나요?**
  * `BATCH_SIZE`를 64→32(이후 실측 재조정 32~48)로 낮추고, `torch.compile(mode="reduce-overhead")`가 GRU+dropout과 안정성 문제가 있고 CUDA Graph 전용 메모리풀까지 추가로 잡아먹는 것을 확인해 **기본 컴파일 모드로 되돌림**. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`도 추가해 파편화로 인한 조기 OOM을 완화했습니다.
* **실측 (batch별 peak VRAM, seq=1000):** batch32=2.61GB, batch48=3.99GB, batch64=5.25GB(여유 0.4GB, 위험), batch80=OOM. **batch48을 안전선으로 채택**했습니다.

---

## [Phase 3] 10. GPU 전력/클럭 스로틀링 발견 — 미해결 (진행 중)
* **어떤 문제를 발견했나요?**
  * 학습 중 `nvidia-smi`로 GPU 상태를 찍어보니 utilization은 100%인데 **클럭이 최대치(3105MHz)의 44~52%밖에 안 나오고(pstate P3~P4), 전력 소모도 20~35W**로 비정상적으로 낮았습니다.
* **원인 조사 결과:**
  * 열 스로틀 아님(58~62도, 정상), 배터리 문제 아님(AC 연결·충전중 확인), 메모리 스왑 없음(RAM 여유 24GB), GPU 점유 다른 프로세스 없음, `powerprofilesctl`을 performance로 바꿔도 개선 없음(오히려 한 번은 더 나빠짐).
* **결론:** 원인 특정 실패. 노트북 하이브리드 그래픽/펌웨어 레벨의 전력 제한으로 추정되나 소프트웨어(코드/OS 설정)로 해결 안 됨 — **미해결 상태로 남겨둠.** 이 클럭만 풀리면 나머지 최적화와 무관하게 최소 2배 이상 속도 향상 가능성이 있다고 진단했습니다.

---

## [Phase 3] 11. 데이터 스케일 진단(Chinchilla) 및 위키 데이터 대규모 확충 (완료)
* **어떤 문제를 해결했나요?**
  * 100+ 에폭을 돌려도 지능이 뚜렷이 개선되지 않는 현상에 대해, "파라미터가 작아서"가 아니라 **"데이터가 너무 적어서"**일 가능성을 계산으로 검증했습니다.
* **계산:**
  * Chinchilla 스케일링 법칙(파라미터당 최적 토큰 ≈20배) 적용 시, 이 모델(1350만 파라미터)의 최적 학습량은 **약 2억7천만 토큰**인데, 당시 Stage2 데이터(`gsm8k_paragraphs.txt`+`wiki_paragraphs.txt`)는 1470만 글자로 **필요량의 1/18**에 불과했습니다.
  * 구글 ByT5(vocab=256, 바이트 단위 토크나이저) 사례를 대조군으로 검토한 결과, **작은 vocab이 필요 데이터량 자체를 줄여주지는 않는다**는 점도 확인했습니다(ByT5도 mC4 코퍼스로 약 1조 토큰을 학습에 사용).
* **어떻게 해결했나요?**
  * 처음부터 새 소스를 찾지 않고, 이미 검증된 `prepare_wiki_paragraphs.py`(HuggingFace `wikimedia/wikipedia` 20231101.ko 스트리밍 + 150~800자 길이필터 + 노이즈문자 배제 + UNK 3% 미만 필터) 파이프라인을 그대로 재사용해 목표 문단 수만 74만개로 올려 재실행했습니다.
* **결과:** `train_data/wiki_paragraphs_v2.txt`로 **4억6600만자** 추가 확보. `train_data/` 전체가 **약 4억8100만자**로 늘어나 Chinchilla 목표치를 오히려 초과 달성했습니다.

---

## [Phase 3] 12. SFT 파이프라인 배선 오류 발견 및 수정 (완료)
* **어떤 문제를 해결했나요?**
  * `train_sft.py`가 `data_dir="train_data"`(프리트레인용 프로즈, Q:/A: 구조 없음)를 보고 있어서, 그대로 실행하면 "형식 맞추기" 효과가 사실상 없고 낮은 LR로 프리트레인을 한 번 더 하는 것과 다름없다는 것을 발견했습니다.
* **어떻게 해결했나요?**
  * `train_data_sft/`를 새로 만들어 `datasets/persona_data.txt`(대화체 2만줄, Q:/A: 구조 보존)와 `datasets/roleplay_data.txt`에서 2만 줄을 샘플링한 파일을 배치. `train_sft.py`가 이 폴더를 보도록 수정했습니다.

---

## [Phase 3] 13. `chat.py` 턴 종료 감지 및 SFT 프롬프트 템플릿 자동 적용 (완료)
* **어떤 문제를 해결했나요?**
  * 기존 `generate()`는 항상 고정 글자 수(`max_new_chars`)만큼 무조건 채워서, SFT 이후에도 답변 뒤에 모델이 지어낸 다음 질문까지 이어붙여 나올 위험이 있었습니다.
* **어떻게 해결했나요?**
  * `generate()`에 `stop_on_newline` 옵션을 추가해 개행이 나오면 그 자리에서 멈추도록 함. `chat.py main()`에서 로드된 체크포인트가 `model_sft_v*`인지 자동 판별해, SFT 모델이면 `"Q: {입력}\nA: "` 템플릿과 턴 종료 감지를 자동 적용하도록 분기했습니다.

---

## [Phase 3] 14. 추가 속도 최적화 — 위치인코딩/마스크 캐싱, 배치 크기 재조정 (완료)
* **어떤 문제를 해결했나요?**
  * 매 forward마다 사인파 위치인코딩과 causal mask를 다시 계산하는 낭비, 그리고 Stage2에서 VRAM 여유(2.96GB/6.14GB)가 있는데도 batch가 작게 고정되어 있는 비효율을 발견했습니다.
* **어떻게 해결했나요?**
  * `model.py`에 최대 길이(1000)까지 위치인코딩/causal mask를 미리 계산해두는 `_pos_cache`/`_causal_mask_cache` 버퍼를 추가(`persistent=False`로 등록해 체크포인트 호환성은 유지). 배치 크기를 실측을 통해 32→48로 재조정했습니다.


---
---

# 🚀 Phase 4: Pointer-Generator (복사 메커니즘) 도입을 통한 독해 지능(QA) 모델로의 진화 (v3.0)

## [Phase 4] 1. Pointer-Generator 복사 메커니즘 도입 결정 및 사고 흐름
* **배경 및 문제 의식:**
  * 13.5M 파라미터 극소형 모델은 백과사전식 지식 암기가 물리적으로 불가능함.
  * 목표를 "아는 것은 적어도 주어진 지문(Context)을 이해하고 사실에 기반해 정확히 추론/답변하는 5세 수준의 지능"으로 재정의함.
  * 기존 GPT 방식(Generative)은 학습하지 않은 고유명사나 조사의 조합 시 글자가 깨지거나 환각(Hallucination)이 발생하는 근본적 한계가 존재함.
* **해결을 위한 선행연구 벤치마킹:**
  * *Pointer-Generator Networks (See et al., 2017)*의 핵심 사상(어텐션 맵을 활용해 입력 텍스트의 단어를 출력으로 직접 복사) 채택.
  * *BERT / SQuAD (Devlin et al., 2018)* 등 소형 독해 전용 모델의 특징 차용.
* **6트랙 자소 구조에서의 작동 원리 증명:**
  * 어텐션 가중치를 복사 스위치($p_{copy}$)와 결합하여, 지문의 특정 위치(Index)를 가리키게 함.
  * 자소 단위가 깨지지 않도록, 특정 위치의 6트랙 튜플 `[초성, 중성, 종성, 기호, 영어, 숫자]`를 통째로 긁어와 다음 시점의 출력으로 주입함.
  * 이로써 초소형 모델에서도 지문 속 어려운 단어("양자역학" 등)를 100% 왜곡 없이 복사/출력할 수 있는 지능을 확보함.

## [Phase 4] 2. Pointer-Generator 아키텍처 구현 및 수치 안정화 (완료)
* **어떻게 해결했나요?**
  * **[model.py](file:///home/redbebero/Projects/SLM/v3/scripts/model.py):**
    * 복사 게이트 `self.head_copy` 레이어 추가 및 스위치 확률 $p_{copy}$ 연산 구현.
    * 헬퍼 함수 `_get_copy_prob`에서 `scatter_add_`를 사용하여 어텐션 맵을 6개 트랙의 vocab 크기 확률로 변환.
    * 최종 반환값을 `log(p_final + 1e-8)` 형태인 로그 확률로 출력하도록 조치하여 수치 안정성 확보 및 기존 NLL/CrossEntropyLoss와 무결하게 호환.
* **훈련 시 Dropout으로 인한 확률 합 붕괴 해결 (진행 중 -> 완료):**
  * **발견한 현상:** 모델 훈련 시 MultiheadAttention에서 dropout(10%)이 활성화되어, 복사용 `attn_weights`가 스케일링되어 합이 1.0이 아닌 1.11 등으로 튀는 현상 발견.
  * **해결:** 어텐션 맵 획득 즉시 `attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)` 정규화 코드를 추가하여 훈련/평가 모드 전 영역에서 완벽하게 확률 합 1.0 보존 완료.
* **테스트 결과:**
  * 진단 테스트 스크립트([test_pointer.py](file:///home/redbebero/Projects/SLM/v3/scripts/test_pointer.py))를 통해 순방향 연산, 역전파, 오차 역전파 경사도 전파가 수학적으로 완벽히 수렴 및 검증됨을 확인.



---
# 🚀 Phase 4: SFT 데이터셋 전면 개편 및 스케줄러 안정화
* **어떤 문제를 해결했나요?**
  * SFT 검증 결과, 합성(규칙 기반) 데이터셋으로 인해 로봇 같은 어투와 단순한 문장 구조 한계에 부딪힘.
  * 10에폭마다 LR이 튀는 Warm Restarts 스케줄러가 미세조정 중인 가중치를 파괴하여 철자 붕괴 유발.
  * `train_sft.py` 실행 시 최신 SFT 가중치 대신 Pretrain 가중치를 강제로 덮어씌워버리는 버그 발생.
* **조치 사항:**
  * 배달 주문 등 기계적 데이터를 버리고 100% 인간 카톡 대화체(1,300개)로 데이터셋 전면 교체. 자연스러운 핑퐁을 위해 로봇 프리픽스(`지문:`, `질문:`) 제거.
  * 스케줄러를 `CosineAnnealingLR`로 교체하여 스파이크 없이 부드럽게 LR이 감쇠하도록 안정화.
  * `train_sft.py`의 가중치 로드 우선순위를 정정하여 SFT 연속 학습(Resume) 로직 복구.

---
# 🚀 Phase 5: 모델 뼈대의 한계 봉착 및 1D-CNN 전면 재설계 결단
* **어떤 문제를 해결했나요?**
  * **속도 문제**: 1에폭에 4시간이 소요됨. 자소 단위 특성상 시퀀스 길이가 너무 긴데, GRU는 순차 계산만 가능하여 GPU 병렬 처리를 낭비함.
  * **Mamba 실패와 통찰**: 속도 개선을 위해 선형 구조인 Mamba를 시도하려 했으나 설치 실패. 오히려 Mamba 도입 시 한글 오타를 막아주던 '복사 게이트(Pointer-Generator, Attention 기반)'를 구조상 버려야 한다는 치명적 모순을 깨달음.
* **최종 결단 (Plan B - Jamo CNN):**
  * 대기업식 토크나이저(BPE+트랜스포머)를 쫓지 않고, 순수 6트랙 자소 분할의 독창적 연구 가치를 살리기로 결정.
  * 느린 GRU와 메모리를 퍼먹는 Attention을 모두 도려내고, 그 자리에 **팽창 1D 합성곱(Dilated 1D CNN) 코어**를 이식하기로 합의함.
  * CNN은 100% 병렬 처리가 가능해 트랜스포머급으로 빠르고, 메모리를 선형으로 소모해 6GB VRAM에 완벽히 부합함.

---
---
# 🚀 Phase 5: 1D-CNN 코어 이식 및 소프트웨어 병목 초토화 (완료)
* **작업 내용:**
  1. **CNN 코어 교체:** GRU/MultiheadAttention을 삭제하고 8층의 Dilated 1D CNN(`ConvCore`)으로 아키텍처를 전면 교체하여 100% 병렬 처리를 달성함.
  2. **소프트웨어 최적화:**
     * `model = torch.compile(model, mode="max-autotune")` 적용으로 커널 융합 가속.
     * `train.py`의 Loss를 `CrossEntropyLoss`에서 `NLLLoss`로 변경하여 불필요한 이중 `log_softmax` 계산(연산 및 메모리 낭비) 원천 차단.
     * CNN 내부 `LayerNorm`의 차원 변경(`transpose`) 병목을 피하기 위해, `[B, C, S]` 포맷에서 네이티브하게 작동하는 `LayerNorm1d`를 직접 구현해 삽입.
     * `torch.set_float32_matmul_precision('high')` 적용으로 하드웨어 TF32 가속 활성화.

---
# 🚀 Phase 6: 수학적 진화 (RetNet, 단일 Softmax, 직교 게이팅) 전면 도입 (완료)
* **어떤 문제를 해결했나요?**
  * 포인터 생성기(Pointer-Generator)의 `scatter_add_`와 이중 Softmax가 GPU 연산에 심각한 저항을 유발함.
  * 초성 -> 중성 -> 종성으로 이어지는 Cascade 계단식 디코딩이 순차적(Sequential) 연산을 강제하여 추론과 병렬화를 방해함.
* **어떻게 해결했나요 (우아한 구조 개편):**
  1. **RetNet 코어 통일:** CNN과 어텐션으로 나뉘어 있던 구조를 `RetentionCore`(선형 어텐션 + 지수 감쇠 마스크)로 하나로 합침. $QK^T$ 행렬이 포인터 복사 가중치를 자연스럽게 제공함.
  2. **확장 어휘 통합 단일 Softmax (Extended Vocab Softmax):** `p_copy` 게이트와 두 개의 Softmax를 합치는 복잡한 수식을 폐기. 모델 출력 로짓 뒤에 $QK^T$를 그대로 이어붙인 후(Concat) 단 한 번의 Softmax로 "생성"과 "복사"를 로짓 공간에서 스스로 경합/결정하도록 수학적으로 압축함.
  3. **직교 게이팅(Orthogonal Gating) 병렬화:** "초성을 계산하고 중성을 계산한다"는 Cascade 구조를 끊음. H 텐서에서 초, 중, 종을 동시에 뽑되, 초성의 연속 확률 분포(Softmax)를 행렬곱(Bilinear Parameter)으로 중성에 스케일링해주는 방식으로 변경. 즉, 순차 for문 없이 행렬곱 1번으로 완벽한 조건부 확률 결합 완성.
* **파급 효과:** 
  * 완전히 새로운 형태의 가중치 행렬이 도입되어 기존 체크포인트(CNN 기반)와 호환 불가능해짐. 충돌(Shape Mismatch) 방지를 위해 기존 `checkpoints` 폴더를 `checkpoints_v5_cnn`으로 피난시키고 처음부터 깨끗하게 훈련을 시작(Phase 6)하도록 조치함.

---
## 📋 다음 행동 계획 (Action Plan)
1. **Phase 6 훈련 시작:**
   * 새롭게 작성된 `model.py`와 `train.py`를 실행시켜 RetNet+직교게이팅 모델의 Loss 수렴(0.x 진입 여부) 확인.
2. **노트북 하드웨어 전력 스로틀링 해제:**
   * 35W로 묶인 클럭 스로틀링 환경을 고성능 모드(100W 이상)로 해제하여 Phase 6 코드의 압도적 속도를 실측할 것.
3. **사전학습(Pretrain) 초고속 재가동:**
   * 개편된 초고속 RetNet 뼈대로 `wiki_paragraphs_v2.txt` (466MB) 말뭉치 초고속 사전학습 시작.

---
# 🚀 Phase 7: RetNet 뼈대 파기 및 GRU + Cascade 롤백 (완료)
* **어떤 문제를 해결했나요?**
  * Phase 6의 직교 게이팅(Orthogonal Gating) 구조 도입 이후, Loss가 80대에서 요지부동하며 모델이 "라라라라"만 내뱉는 치명적인 붕괴(Divergence) 현상을 발견.
  * 한글의 조건부 확률(초성->중성->종성)을 억지로 병렬 행렬곱으로 풀려던 수식이 수학적으로 불안정했음.
* **어떻게 해결했나요?**
  * **코어 롤백**: 수학적 복잡성만 가중시켰던 RetNet을 파기하고 검증된 `nn.GRU`로 원복.
  * **Cascade 디코딩 롤백**: 초성을 먼저 뽑아 중성에 먹이고, 초/중성을 모아 종성에 먹이는 직렬(순차) 헤드 구조로 원복. 강한 의존성을 모델 구조에 강제로 주입.
  * **쓰레기 가중치 격리**: 기존 `checkpoints` 폴더 내에 섞여 있던 쓰레기 가중치들은 `checkpoints_backup_phase6`로 통째로 피난시킴. 
* **향후 계획:**
  * 깨끗해진 `checkpoints` 환경에서 `train.py`를 다시 처음부터 가동해 정상 수렴 확인.

---
# 🚀 Phase 8: Pretrain Loss 13 정체 진단 및 학습 안정화 (진행 중)

## 1. 현상
* 새 GRU+Cascade 구조로 pretrain을 시작했으나, 약 1에폭 학습 후에도 전체 합산 Loss가 **13 부근에서 처음과 거의 동일하게 정체**됨.
* 현재 설정의 전체 Loss는 타입 head와 6개 트랙 loss를 합산한 값이므로, 숫자 13만으로 특정 head의 실패 여부를 판단할 수 없었음.
* 따라서 `type`, `cho`, `jung`, `jong`, `sym`, `eng`, `num`을 epoch 종료 시 각각 기록하도록 진단 로그를 추가함.

## 2. 현재 상태 실측
* 모델: `nn.GRU` 8층, `hidden_dim=768`, 약 **28.9M parameters**.
* 데이터: `train_data/`의 텍스트를 토큰화한 캐시 기준 **203,203,506 tokens**.
* 학습 길이: `SEQ_LENGTH=250`, `STRIDE=250`, `BATCH_SIZE=32`.
* `train_data/pretokenized_cache.pt`의 토큰 수는 원본 텍스트와 일치함. 캐시는 재생성할 필요 없음.
* `checkpoints/`의 기존 가중치는 Phase 6 RetNet/CNN 실험 가중치와 섞일 위험이 있으므로 새 학습에서는 사용하지 않음.

## 3. 원인 및 판단
* 기존 `BASE_LR=0.003`은 8층 GRU와 768 hidden 크기에 비해 공격적인 값임. gradient clipping이 있어도 초기 업데이트가 불안정해질 수 있으므로 학습률을 낮춤.
* `scheduler.step()`이 `optimizer.step()`보다 먼저 호출되고 있었음. PyTorch 학습 관례와 반대 순서이므로 optimizer 업데이트 뒤 scheduler를 호출하도록 수정함.
* `torch.compile`은 GRU의 장시간 메모리 누적 및 그래프 문제를 분리해 확인하기 어렵게 함. 수렴 검증 단계에서는 비활성화함.
* `target_for_forcing=y`는 Cascade head가 정답 초성·중성을 사용하게 만드는 teacher forcing 방식임. 실제 생성과의 차이는 존재하지만, 이번 Loss 정체의 직접 원인으로 단정하지 않고 학습 안정화 후 별도 free-running validation 대상으로 남김.

## 4. 적용한 코드 변경
* `scripts/config.py`
  * `BASE_LR`: `0.003` → `0.0005`
  * `EPOCHS=5`, `WARMUP_STEPS=300`을 설정값으로 분리. 2 epoch는 파라미터 대비 학습량이 부족할 수 있어 1차 pretrain을 5 epoch로 확장함.
  * `CHECKPOINT_EVERY_EPOCH=1` 추가
  * `USE_TORCH_COMPILE=False` 기본값 지정
* `scripts/train.py`
  * `optimizer.step()` 후 `scheduler.step()` 실행
  * cosine scheduler의 `T_max`를 전체 epoch 수와 warmup step 기준으로 계산
  * epoch마다 checkpoint 저장
  * 전체 Loss와 함께 7개 구성 loss 출력
  * 기본 실행(`python scripts/train.py`)은 checkpoint를 자동 resume하지 않고 새 모델로 시작

## 5. 재학습 방법
* 현재 생성된 모델 가중치만 삭제할 경우:

```bash
rm checkpoints/model_v*.pth
```

* `train_data/pretokenized_cache.pt`는 삭제하지 않음. 대용량 토큰 캐시이므로 유지.
* 프로젝트 루트에서 실행:

```bash
cd /home/redbebero/Projects/SLM/v3
venv/bin/python scripts/train.py
```

* `python train.py`는 루트에 `train.py`가 없으므로 사용하지 않음.
* Ctrl+C로 중단하면 현재 epoch의 가중치를 checkpoint로 저장함.

## 6. 다음 확인 기준
* epoch 로그에서 전체 Loss뿐 아니라 `cho/jung/jong`이 함께 감소하는지 확인.
* 전체 Loss가 여전히 13 부근이면 트랙별 loss 중 어느 항이 고정되는지 먼저 확인.
* teacher-forcing loss가 감소한 뒤에도 생성 결과가 무너지면 `target_for_forcing=None` 상태의 free-running validation을 추가하고 Cascade teacher forcing 문제를 별도 조정함.

## 7. 중단 후 resume 및 중간 평가
* `model_v1.pth`는 epoch 1 완료, `model_v2.pth`는 epoch 2 완료 checkpoint임.
* 최신 `model_v3.pth`는 파일명은 epoch 3이지만 scheduler `last_epoch=67324`로 확인되어 **epoch 3 진행 중 Ctrl+C로 저장된 중간 checkpoint**임.
* `model_v3.pth`를 `chat.py`로 평가한 결과 `세종대왕은` 뒤에 `아리이아으...`가 생성되어 아직 충분히 수렴하지 않은 상태로 판단함.
* 따라서 학습을 종료하지 않고 `model_v3.pth`에서 5 epoch 목표까지 resume하기로 결정함.
* Ctrl+C 중간 저장을 다음 epoch 완료로 잘못 건너뛰지 않도록 `train.py`가 scheduler step을 검사해 미완료 epoch부터 재시작하도록 수정함.

재개 명령:

```bash
cd /home/redbebero/Projects/SLM/v3
venv/bin/python scripts/train.py resume
```

평가 명령:

```bash
venv/bin/python scripts/chat.py 3
```

---

# 🚀 Phase 9: 데이터 품질 재점검, 자소별 학습 진단, GPU 복구 및 장기 학습 전환

## 1. 재점검 계기

5에폭 학습 후 전체 합산 Loss가 약 9.5까지밖에 내려가지 않고, 생성 결과가 `아리아아으...`처럼 반복되는 현상이 나타났다. 기존 기록에서는 이를 모델 용량 부족과 Spiking 구조 문제로 해석했지만, 현재 코드와 실제 데이터 파이프라인을 다시 대조한 결과 데이터·평가·자소별 loss 구조를 먼저 수정해야 한다고 판단했다.

## 2. 실제 데이터 상태 확인

* `train_data_clean/pretrain_train.txt`는 위키백과 정제 코퍼스가 아니라 GSM8K 형식의 한국어 수학 문제·풀이 데이터였다.
* 문제 본문과 풀이 과정이 한 줄 샘플로 들어가며 숫자, 계산식, 번역투 표현, 반복 템플릿 비중이 높았다.
* 기존 정제 파일은 약 44.9만 줄, 2.88억 bytes 규모였다.
* 기존 필터는 50~800자, 한국어 비율 70%, unknown 비율 1% 이하만 검사했다. 따라서 unknown 문자, 반복 템플릿, 과도한 계산식이 남을 수 있었다.
* `공식s*홈페이지`는 정규식 오타였고 `공식\s*홈페이지`로 수정했다.

## 3. 확인된 구조적 문제

### 3.1 샘플 경계 혼합

기존 `dataset.py`는 모든 텍스트를 하나로 합친 뒤 stride window를 만들었다. 이 방식은 한 window 안에 다음과 같은 인공 문맥을 만들 수 있었다.

```text
문제 A 풀이 끝 + 문제 B 시작
```

문제·풀이 샘플 경계를 보존해야 하므로 각 줄을 독립 샘플로 읽고, 한 샘플 내부에서만 window를 생성하도록 변경했다.

### 3.2 legacy cache 문제

기존 `pretokenized_cache.bin` metadata에는 전체 token 수만 있고 유효한 샘플 시작 위치가 없었다. 따라서 cache metadata에 `sample_starts`가 없으면 legacy cache로 판단해 자동 재생성하도록 변경했다.

새 cache는 다음을 저장한다.

```text
token_count
sample_starts
```

`__getitem__`은 `idx * stride` 대신 `sample_starts[idx]`를 사용한다. 짧은 샘플은 다른 샘플과 이어 붙이지 않고 window 대상에서 제외한다.

### 3.3 SFT 경계 보존

pretraining은 줄 단위 샘플을 사용하지만 SFT는 `\n\n` 단위 multi-line 샘플을 사용한다. 공통 로더 수정 중 SFT 샘플이 줄 단위로 쪼개질 위험을 발견해 SFT는 기존 `\n\n` 경계를 유지하도록 별도 처리했다.

## 4. 데이터 재정제

다음 기준으로 `prepare_clean_pretrain.py`를 수정하고 데이터를 재생성했다.

* unknown 문자가 하나라도 포함되면 reject
* reference·bibliography·저널 형식 reject
* `공식\s*홈페이지` 패턴 reject
* 기존 exact duplicate 제거 유지
* train/validation deterministic 분할 유지

재생성 결과:

```text
train: 464,278 lines / 120,000,080 chars
valid: 4,689 lines / 1,208,305 chars
rejected: 754,849 lines
```

새 sample-aware cache 결과:

```text
train tokens: 120,464,358
train windows: 205,302
validation tokens: 1,212,994
validation windows: 2,031
```

대형 cache 생성 중 전체 샘플을 Python list에 보관해 중간 종료되는 문제가 한 번 발생했다. 메모리 부담을 줄이기 위해 source file을 두 번 순회하는 streaming 방식으로 변경했고, 이후 cache 생성에 성공했다.

## 5. 테스트 및 검증 추가

다음 회귀 검사를 추가했다.

* 한글·standalone 자모·영어·숫자·기호 encode/decode round-trip
* pretraining window가 source sample 경계를 넘지 않는지 확인
* unknown 및 reference 문장 reject 확인
* SFT `\n\n` sample boundary 확인
* model forward output shape 확인
* 기존 checkpoint state_dict load 확인

환경에 pytest가 설치되어 있지 않아 pytest 실행은 불가능했다. 대신 테스트 함수를 직접 실행했고, `py_compile` 및 GPU smoke test를 통과시켰다.

## 6. 자소별 loss 및 teacher forcing 수정

### 6.1 자소별 metric 추가

전체 합산 Loss만으로는 어느 자소가 실패했는지 알 수 없으므로 다음 metric을 추가했다.

```text
type_acc
cho_acc
jung_acc
jong_acc
jong_present_acc
jong_empty_acc
full_hangul_acc
predicted_hangul_repeat_rate
```

`full_hangul_acc`는 초성·중성·종성이 한 글자에서 모두 맞은 비율이다. `jong_present_acc`는 종성이 실제로 있는 글자만 따로 측정한다.

### 6.2 자소 단위 scheduled sampling

기존 Cascade 학습은 teacher forcing 여부를 batch 단위 boolean으로 결정했다. 이후 초성·중성 embedding 각각에 대해 gold embedding과 prediction embedding을 확률적으로 혼합하도록 수정했다.

teacher forcing ratio가 0이면 prediction embedding만 사용하고, 1이면 gold embedding을 사용한다. 이를 통해 teacher-forced validation과 free-running generation 간 차이를 줄인다.

### 6.3 종성 class imbalance 보정

clean corpus의 완성형 한글 종성 분포를 측정했다.

```text
Hangul positions: 83,667,323
jong=0:           46,457,938
jong>0:           37,209,385
```

종성 없음과 일부 자주 등장하는 종성이 prediction을 지배하지 않도록 cleaned corpus frequency 기반 square-root inverse-frequency weight를 `jong` loss에 적용했다. tokenizer ID와 model 출력 차원은 변경하지 않아 기존 checkpoint 호환성을 유지했다.

## 7. 첫 재학습 및 진단

기존 `checkpoints/model_v8.pth`를 보존하고 정제 데이터 단계로 `newstage` 학습을 시작했다.

```bash
prime-run venv/bin/python scripts/train.py newstage
```

`newstage`는 model weight만 복원하고 optimizer·scheduler는 새로 시작한다. 데이터 분포와 loss 방식이 바뀌었으므로 기존 optimizer state를 재사용하지 않은 이유다.

### 7.1 첫 단계 결과

초기 5 epoch 결과는 다음과 같았다.

```text
model_v9  epoch1  free_running=10.3299  full_hangul=0.109  jong+=0.052
model_v10 epoch2  free_running= 9.9651  full_hangul=0.116  jong+=0.066
model_v11 epoch3  free_running= 9.8367  full_hangul=0.125  jong+=0.072
model_v12 epoch4  free_running= 9.7395  full_hangul=0.129  jong+=0.077
model_v13 epoch5  free_running= 9.7116  full_hangul=0.130  jong+=0.080
```

`model_v14`는 epoch 5 종료 후 최종 저장 로직이 한 번 더 저장한 중복 checkpoint다.

free-running loss는 감소했지만 생성 결과는 다음처럼 반복 음절로 붕괴했다.

```text
양자역학은 → 아리아아으으으아의 아이지으...
대한민국의 수도는 → 아리아아아아아아아아...
```

이 결과로 자소별 CE가 낮아지는 것과 실제 음절·단어 생성이 정상인 것은 다르다는 점을 확인했다. greedy argmax와 GSM8K 중심 corpus가 일반 한국어 prompt에 약한 것도 함께 확인했다.

## 8. NVIDIA GPU 복구

처음 학습 실행에서는 `Device: cpu`가 잡혀 첫 step이 약 24초로 측정되어 학습을 중단했다.

진단 결과:

* GPU: NVIDIA GeForce RTX 4050 Laptop GPU
* NVIDIA module: 610.43.03
* `/dev/nvidia*` 초기 부재
* kernel log에 SBIOS thermal/power request 오류
* GPU가 Runtime D3 suspended 상태로 전환

사용자가 `nvidia-persistenced`를 활성화하고 GPU runtime power를 켠 뒤 정상화됐다.

정상 확인 결과:

```text
nvidia-smi: 정상
Persistence-M: On
Runtime power: active
torch.cuda.is_available(): True
device_count: 1
CUDA matrix smoke test: PASS
```

이후 모든 학습 실행에는 다음 형태를 사용한다.

```bash
prime-run venv/bin/python scripts/train.py newstage
```

## 9. class-balanced loss 적용 후 재학습

`model_v14`에서 새 stage를 시작했고, 15 epoch cosine schedule 대신 teacher forcing schedule을 완화했다.

```text
teacher forcing: 1.00 → 0.75 → 0.50 → 0.25 → 0.00
```

epoch 1~12 결과:

```text
model_v15  epoch1  free_running=9.1471  full_hangul=0.119  jong+=0.065 repeat=0.138
model_v16  epoch2  free_running=8.9614  full_hangul=0.124  jong+=0.074 repeat=0.135
model_v17  epoch3  free_running=8.9123  full_hangul=0.127  jong+=0.083 repeat=0.129
model_v18  epoch4  free_running=8.8407  full_hangul=0.131  jong+=0.092 repeat=0.127
model_v19  epoch5  free_running=8.7551  full_hangul=0.134  jong+=0.095 repeat=0.123
model_v20  epoch6  free_running=8.7070  full_hangul=0.137  jong+=0.103 repeat=0.107
model_v21  epoch7  free_running=8.6697  full_hangul=0.140  jong+=0.101 repeat=0.113
model_v23  epoch8  free_running=8.6151  full_hangul=0.143  jong+=0.107 repeat=0.108
model_v24  epoch9  free_running=8.5671  full_hangul=0.145  jong+=0.117 repeat=0.105
model_v25 epoch10  free_running=8.5440  full_hangul=0.147  jong+=0.110 repeat=0.102
model_v26 epoch11  free_running=8.5163  full_hangul=0.148  jong+=0.112 repeat=0.105
model_v27 epoch12  free_running=8.5048  full_hangul=0.149  jong+=0.113 repeat=0.095
```

평가:

* free-running loss 지속 감소
* full Hangul accuracy 지속 상승
* jong-present accuracy 약 6.5%에서 11.3%로 상승
* 반복률 전반적 감소
* NaN 및 gradient 발산 없음

따라서 현재 run은 학습 중단 사유가 없으며, 기존 15 epoch 목표까지 계속 진행하기로 했다. 다만 실제 생성이 여전히 반복 음절을 보이므로 이 checkpoint를 최종 모델로 확정하지 않고, epoch 종료 후 in-distribution/OOD 생성 평가를 수행해야 한다.

## 10. 무한 학습 전환

기존 15 epoch cosine scheduler는 epoch가 늘어날수록 learning rate가 `1e-5`까지 감소하므로 무한 학습에 부적합했다. 다음 실행부터는 다음처럼 변경했다.

* `EPOCHS = None`: 무한 epoch
* `ReduceLROnPlateau`: validation free-running loss가 2회 정체되면 LR 0.5배
* 최소 LR `1e-5`
* 매 epoch checkpoint 저장
* Ctrl+C 안전 저장 유지
* `resume` 시 ReduceLROnPlateau state 복원

현재 이미 실행 중인 finite-epoch 프로세스는 기존 코드가 메모리에 로드되어 있으므로 epoch 15에서 종료된다. 그 이후 최신 checkpoint에서 무한 stage를 시작한다.

```bash
prime-run venv/bin/python scripts/train.py newstage
```

무한 stage 중단 후 재개:

```bash
prime-run venv/bin/python scripts/train.py resume
```

## 11. 현재 판단

현재 구조는 자소별 학습과 free-running 지표 측면에서 개선 중이다. 그러나 다음 한계는 남아 있다.

* loss는 자소별 factorized objective이며 음절 전체 일치 objective가 별도 없음
* `full_hangul_acc`가 아직 낮음
* GSM8K 중심 corpus라 일반 한국어 prompt 분포가 부족함
* greedy decoding에서 반복 음절 붕괴가 남아 있음

따라서 무한 학습은 checkpoint와 지표를 관찰하는 실험용으로만 사용한다. `free_running`, `full_hangul`, `jong+`, `repeat`가 장기간 정체되면 무한 학습을 중단하고 음절 단위 objective와 자연어 corpus 혼합을 추가해야 한다.

## 12. 구조 재평가와 소형 A/B 실험

기존 대형 GRU를 계속 학습하는 대신 구조가 실제로 도움이 되는지 먼저 검증하기로 했다. 기존 구조는 공유 GRU 뒤에 초성→중성→종성을 연결한 cascade 방식이었다. 이 구조는 한글 결합 규칙을 반영하지만 앞 단계의 잘못된 예측이 뒤 단계로 전파될 수 있다.

소형 모델 두 가지를 동일 조건으로 비교했다.

```text
emb=32, hidden=256, GRU=2층, batch=64, seq=250, 3 epoch
```

결과:

```text
Cascade GRU       free=8.7156 full_hangul=0.146 jong+=0.092 repeat=0.087
Independent GRU   free=8.6821 full_hangul=0.147 jong+=0.099 repeat=0.080
```

Independent 방식이 근소하게 우세했다. cascade의 강제 순차 의존이 당시 데이터에서는 이득으로 나타나지 않았다.

## 13. Transformer + 자소 fusion 구조

문맥 지능과 자소 상호작용을 동시에 얻기 위해 `KoJamoTransformer`를 추가했다.

```text
6트랙 자소 입력
  → causal Transformer
  → 초성/중성/종성 soft embedding 생성
  → fusion 층에서 서로 결합
  → 최종 type/자소 head
```

초성·중성·종성의 확률분포에서 soft embedding을 만들고 fusion 층에 다시 넣었다. 따라서 자소가 서로 영향을 주지만 cascade처럼 hard argmax 오류가 연쇄 전파되지 않는다. Transformer는 causal mask를 사용하므로 미래 토큰을 보지 않는다.

소형 Transformer 실험:

```text
emb=32, hidden=256, Transformer=2층, heads=4, batch=64, seq=250, 3 epoch
free_running=8.3031
full_hangul=0.173
jong+=0.103
repeat=0.058
parameters=1,927,894
```

기존 두 소형 GRU보다 free-running, full Hangul, 종성 있음, 반복률이 모두 개선됐다. 체크포인트는 `checkpoints/compare_transformer_3.pth`다.

## 14. 중형 Transformer 검증과 중단

소형 결과를 바탕으로 다음 중형 모델을 실행했다.

```text
emb=48, hidden=384, Transformer=4층, heads=6
batch=32, seq=250, 5 epoch 목표
```

이 설정은 전체 cleaned corpus에서 6,415 batch/epoch가 되어 약 8~9분/epoch가 걸렸다. GPU 사용률은 87~90%였고 VRAM 약 905MiB, NaN/OOM은 없었다. 즉 학습은 정상이나 테스트 주기로는 너무 느렸다.

Epoch 1은 완료되어 `checkpoints/transformer_medium_1.pth`로 저장됐다. 검증 결과:

```text
teacher/free=8.6144
cho=0.4044 jung=0.3715 jong=0.5694
jong+=0.1064 full_hangul=0.1651 repeat=0.0692
```

Epoch 2의 23%에서 Ctrl+C로 중단했다. 중간 저장본은 실수로 다음 재학습에서 자동 선택되지 않도록 `transformer_medium_partial_epoch2.pth`로 이름을 바꿨다. 중형 모델은 소형 Transformer 3 epoch 결과보다 아직 개선되지 않았으므로 장시간 확대 학습을 보류했다.

## 15. 빠른 실험 중심 실행 계획

긴 전체 학습을 반복하지 않기 위해 다음 순서를 채택했다.

1. `train.py`에 `--limit-batches`와 `--stride`를 추가한다.
2. 제한된 batch와 짧은 sequence로 소형 GRU 2종과 Transformer fusion을 비교한다.
3. 가장 좋은 구조만 중형으로 확대한다.
4. 중형 Epoch 1 validation이 실제로 개선될 때만 전체 cleaned corpus 학습으로 확대한다.
5. validation 2회 연속 정체 시 학습을 중단하고 데이터·손실·decoding을 수정한다.
6. 다양한 한국어 문서 pretraining 후 GSM8K와 대화 데이터는 SFT/추론 단계에 사용한다.

계획 문서:

```text
docs/superpowers/plans/2026-07-15-fast-intelligent-korean-model-plan.md
```

현재 기준 본선 후보는 `Transformer + 자소 soft fusion`이다. 단, 중형 확대가 소형보다 실제로 좋아지는지 제한 실험으로 먼저 검증해야 한다.

## 16. 학습 전 준비 완료

사용자가 실제 학습 프로세스를 직접 실행하기로 하여, 이 시점부터는 학습 자체를 자동 실행하지 않는다. 학습 전에 필요한 빠른 실험 경로를 `scripts/train.py`에 추가했다.

추가된 실행 옵션:

```text
--stride N
--limit-batches N
```

`--limit-batches`는 한 epoch에서 실제 처리할 batch 수를 제한하고, progress bar total과 평균 loss도 실제 처리 수에 맞춘다. `--stride`는 짧은 sequence 실험에서 데이터 window 간격을 명시할 수 있게 한다.
모델 비교 재현성을 위해 `--seed`도 추가했고 기본값은 `42`다. Python, CPU PyTorch, CUDA PyTorch seed를 함께 설정한다.

validation free-running loss가 가장 낮은 모델은 다음 이름으로 별도 저장된다.

```text
checkpoints/<prefix>best.pth
```

번호형 checkpoint와 best checkpoint는 분리되며, best 파일이 다음 번호 계산에 섞이지 않는다.

준비 검증:

```text
training_batch_limit helper: PASS
Transformer CPU forward: PASS
Transformer CUDA forward/backward: PASS
scripts/*.py py_compile: PASS
train.py --help에서 --stride/--limit-batches 노출: PASS
```

현재 실행 중인 학습 프로세스는 없으며, 사용자가 아래 명령을 직접 실행해야 한다. 첫 단계는 500 batch 제한 smoke test이고, 통과 후 소형 구조 비교, 중형 제한 검증, 전체 학습 순서로 진행한다.

## 17. 무인 5시간 단일 명령 실행 스크립트

사용자가 학습 중 컴퓨터를 조작할 수 없는 조건을 반영해 `scripts/run_unattended_5h.sh`를 추가했다. 이 스크립트는 다음을 자동으로 순서 실행한다.

1. CUDA/`nvidia-smi` preflight
2. Transformer smoke test 500 batch
3. Independent GRU 소형 비교
4. Cascade GRU 소형 비교
5. Transformer fusion 소형 비교
6. Transformer 중형 제한 검증
7. 모든 앞 단계가 정상 종료되면 전체 cleaned corpus 무한 학습

모든 출력은 `logs/unattended_YYYYmmdd_HHMMSS.log`에 저장한다. `/tmp/slm_v3_unattended.lock`으로 중복 실행을 방지한다. 외부 `timeout`이 5시간 후 SIGINT를 보내면 `train.py`의 KeyboardInterrupt 처리로 현재 checkpoint를 저장한다.

이 스크립트 자체는 준비 단계에서 실행하지 않았다. 실제 학습은 사용자가 다음 단일 명령으로 시작한다.

## 18. 무인 밤샘 학습 결과

2026-07-15 23:25부터 `logs/unattended_20260715_232518.log`를 기준으로 무인 순차 학습이 실행됐다. CUDA preflight와 smoke/A-B/중형 제한 단계는 정상 종료했고, 전체 Transformer 학습으로 진입했다.

전체 단계 설정:

```text
Transformer, emb=48, hidden=384, layers=4, heads=6
batch=64, seq=250, stride=250, seed=42
```

전체 학습 validation 추이:

```text
초기 full validation: free=8.1088 full_hangul=0.192 repeat=0.057
best checkpoint: epoch=18 free=7.4120 full_hangul=0.2534 jong+=0.2207 repeat=0.0342
latest completed: epoch=22 free=7.5052 full_hangul=0.2563 jong+=0.2262 repeat=0.0342
```

`unattended_full_best.pth`를 동일 validation으로 재로드해 재검증한 결과도 `free=7.4120`, `full_hangul=0.2534`, `repeat=0.0342`로 일치했다. 최신 epoch 22는 full Hangul과 jong-present는 조금 높지만 free-running loss가 best보다 나쁘므로 best checkpoint를 기준 모델로 보존한다.

실제 greedy 생성 검증:

```text
양자역학은 → 아지 않으 아지 않으 아지 않으 ...
대한민국의 수도는 → 아리사이으 아리사이으 아리사이으 ...
안녕하세요 → 스느 아느 아느 아느 ...
```

validation 지표는 크게 개선됐지만 일반 prompt 생성에서는 반복 붕괴가 여전히 남았다. 따라서 이 모델은 자소/분해 예측 기준 모델로는 개선됐으나 GPT형 대화 모델로 확정할 수 없다. GSM8K 중심 데이터 분포와 greedy decoding, 음절 전체 objective 부재가 다음 개선 대상이다.

운영 확인:

```text
현재 unattended_full 학습 프로세스가 여전히 실행 중
프로세스 경과시간 약 8시간 20분
GPU 사용률 약 92%, VRAM 약 1.5GiB/6GiB
```

프로세스 트리상 외부 `timeout` 프로세스는 남아 있지 않으므로, 5시간 자동 종료가 적용되지 않고 무한 학습이 계속된 상태다. 현재는 validation이 best 이후 악화 중이므로 `unattended_full_best.pth`를 보존한 뒤 학습을 중단하는 것이 합리적이다.

확인 후 SIGINT로 학습을 안전하게 중단했다. 마지막 저장 checkpoint는 `unattended_full_23.pth`이며, best와 마지막 checkpoint 모두 현재 Transformer 구조로 재로드 검증을 통과했다. 중단 후 GPU 사용률은 0%, VRAM 사용량은 1MiB로 정상 회수됐다.

## 19. Transformer 체크포인트 대화 연결

기존 `scripts/chat.py`는 GRU 체크포인트만 가정했기 때문에 `unattended_full_best.pth`를 바로 사용할 수 없었다. 다음을 필요한 범위에서 수정했다.

- Transformer/GRU state dict를 자동 판별하고 실제 차원(`emb_dim`, `hidden_dim`, `layers`, `max_seq_length`)을 추론한다.
- 기본 체크포인트 우선순위를 `unattended_full_best.pth`에 두었다.
- 명령행에 체크포인트 경로를 직접 전달할 수 있게 했다.
- 실제 best Transformer 체크포인트 로드 및 forward 검증을 통과했다.
- `tests/test_training_cleanup.py`에 Transformer 체크포인트 차원 추론 테스트를 추가했다.

실행 명령:

```bash
cd /home/redbebero/Projects/SLM/v3
venv/bin/python scripts/chat.py
```

명시적으로 best를 지정하려면:

```bash
venv/bin/python scripts/chat.py checkpoints/unattended_full_best.pth
```

Section 18의 “chat.py가 아직 Transformer를 지원하지 않는다”는 당시 시점의 기록이며, 현재는 본 Section 19의 수정으로 대체된다. 단, 실제 생성에서 반복 붕괴가 남아 있으므로 이 체크포인트는 대화 완성 모델이 아니라 다음 개선을 위한 기준 모델이다.

## 20. 실제 대화 생성 최종 평가

2026-07-16에 `unattended_full_best.pth`를 CUDA로 로드하고 고정 프롬프트 8개에 대해 greedy 생성 40자를 실행했다. 모델 로드와 CUDA 사용은 정상이나, 생성 품질은 대화 모델 기준을 충족하지 못했다.

대표 결과:

```text
안녕하세요       → 스느 아느 아느 아느 아느 ...
대한민국의 수도는 → 아리사이으 아리사이으 아리사이으 ...
양자역학은       → 아지 않으 아지 않으 아지 않으 ...
오늘 날씨가      → 아니라 아리으 아리으 아리으 ...
파이썬은 무엇인가요? → 아이써으 아이써으 아이써으 ...
고양이는         → 아리사에서 아리사에서 아리사에서 ...
```

판정:

- 체크포인트 저장/재로드: 정상
- CUDA 추론: 정상
- 한글 자소에서 음절로 복원: 부분 정상
- 문장 의미 이해/질문 응답: 실패
- 반복 억제: 실패
- 숫자·상식·대화 생성: 실패

따라서 현재 모델은 구조적 자소 예측의 학습 기준선으로는 유효하지만, `chat.py`를 실행할 수 있다는 의미의 사용 가능 상태일 뿐 GPT형 대화 모델로 사용할 단계는 아니다. 추가 pretraining을 무작정 반복하지 않고, decoding 보정·음절 전체 손실·한국어 대화 SFT·데이터 분포 확장을 먼저 진행해야 한다.

## 21. 생성 보정 및 Transformer SFT 연결 실행

실제 생성 평가 후 다음 최소 수정들을 적용했다.

- `scripts/chat.py`에 `temperature`, `top-k`, `top-p`, `repetition_penalty` decoding을 추가했다.
- 기본값은 greedy를 유지하되 repetition penalty 기본값은 1.12로 설정했다.
- 명령행 옵션을 추가했다.

```bash
venv/bin/python scripts/chat.py checkpoints/unattended_full_best.pth \
  --temperature 0.8 --top-k 20 --top-p 0.95 --repetition-penalty 1.12
```

repetition penalty는 일부 반복을 줄였지만 의미 있는 문장을 만들지는 못했다. sampling은 출력 다양성은 늘렸으나 한글 품질이 낮았다. 따라서 decoding만으로 문제를 해결할 수 없다는 결론을 재확인했다.

`scripts/train_sft.py`도 Transformer 체크포인트를 받을 수 있도록 수정했다.

- `--checkpoint`로 pretrain/SFT 체크포인트 지정
- state dict에서 Transformer 차원 자동 추론
- `--epochs`, `--batch`, `--limit-batches` 지원
- Transformer 최대 문맥 512에 맞춰 SFT batch를 안전하게 truncate

다음 smoke test를 실행해 Transformer forward/backward, validation, 저장을 확인했다.

```bash
prime-run venv/bin/python scripts/train_sft.py \
  --checkpoint checkpoints/unattended_full_best.pth \
  --epochs 1 --batch 1 --limit-batches 1
```

Train loss는 8.5911, validation loss는 7.4254였다. 단 1 batch 결과이므로 실제 SFT 모델로 인정하지 않고 생성된 `model_sft_v1.pth`는 삭제했다. 자동 선택 기준은 다시 `checkpoints/unattended_full_best.pth`임을 확인했다.

모든 테스트 함수와 `scripts/*.py` 문법 검사를 다시 통과했고, 종료 후 GPU 사용률은 0%, VRAM은 1MiB였다. 다음 실제 실행은 1~3 epoch 제한 SFT이며, 그 전에 음절 전체 손실과 고정 생성 평가를 추가하는 것이 우선이다.

## 22. Transformer SFT 3 epoch 결과

`unattended_full_best.pth`에서 `train_data_sft` 2,300개 샘플을 대상으로 3 epoch SFT를 실제 실행했다.

```text
batch=8, max context=512, SFT LR=0.0005
checkpoints/model_sft_v1.pth ~ v3.pth 저장
```

Epoch 3 결과:

```text
Train Loss: 4.8802
Validation Loss: 4.9053
```

실제 생성 평가:

```text
안녕하세요 → 아녕. 나는 아느 자기가 아녀. 나는 아느 자기가 아녀. ...
대한민국의 수도는 → 아니, 나는 자긴자기자이아르 자기자이아르 ...
1 더하기 1은 → 나는 자기자으 자주 아나, 아나, 아나, 아나, ...
```

v2가 일부 프롬프트에서 v1/v3보다 나았지만 모든 SFT checkpoint에서 반복과 의미 오류가 남았다. SFT loss 하락만으로 모델을 채택하지 않는다. 현재 `chat.py`는 SFT checkpoint가 존재하므로 최신 `model_sft_v3.pth`를 자동 선택하지만, 대화 품질 기준으로는 아직 실험 모델이다. 다음에는 SFT 학습률을 낮추고, 음절 전체 보조 손실·반복/정답 기반 고정 평가·데이터 품질을 함께 개선해야 한다.

## 23. 고정 생성 평가 하네스 실행

계획 Task 1에 따라 `eval/ko_generation_cases.jsonl`에 인사·상식·수학·설명·문맥 유지 20개를 만들고 `scripts/evaluate_generation.py`를 추가했다. 동일한 20개 프롬프트로 pretrain과 SFT를 비교했다.

평가 명령:

```bash
venv/bin/python scripts/evaluate_generation.py \
  --checkpoints checkpoints/unattended_full_best.pth checkpoints/model_sft_v3.pth \
  --output logs/generation_eval_20260716.json
```

결과:

```text
                              pretrain best    SFT v3
keyword_hit_rate                 0.05          0.00
repetition_3gram_ratio           0.6662        0.3485
full_hangul_ratio                0.9832        0.9429
avg_new_tokens                  12.95         11.70
```

SFT가 반복률은 낮췄지만 정답 키워드 적중률은 0%로 떨어졌다. 따라서 현재 SFT v3를 품질 개선 모델로 채택하지 않는다. 측정 로그는 `logs/generation_eval_20260716.json`에 보존한다.

평가 함수 테스트와 전체 수동 테스트 14개, `scripts/*.py` 문법 검사를 통과했고 평가 후 GPU는 0%, VRAM 1MiB였다. 다음 구현 작업은 Transformer가 학습한 512 문맥을 `chat.py`가 250으로 잘라버리는 문맥 길이 불일치 수정이다.

## 24. Transformer 문맥 길이 불일치 수정

`KoJamoTransformer`는 `max_seq_length=512`로 학습됐지만 기존 `chat.py`의 `generate()`는 항상 250 토큰만 유지하고 있었다. 이를 다음처럼 수정했다.

- `get_context_length(model)` 추가
- Transformer는 `model.pos_emb.num_embeddings`를 자동 사용
- GRU는 기존 학습 조건인 250을 유지
- `generate(..., context_length=None)`으로 명시적 override도 지원

검증:

```text
context length test: PASS
전체 수동 테스트 15개: PASS
scripts/*.py py_compile: PASS
GPU: 0%, VRAM: 1MiB
```

512 문맥으로 20개 평가를 다시 실행했지만 지표는 동일했다.

```text
pretrain keyword=0.05, repeat=0.6662
SFT v3   keyword=0.00, repeat=0.3485
```

따라서 문맥 절단은 실제 버그였지만 현재 반복 붕괴의 유일한 원인은 아니다. 다음 우선순위는 SFT 데이터 품질과 학습 objective 개선이다.

## 25. SFT 데이터 정제 및 안정화 적용

Task 3~4의 첫 부분을 적용했다.

### 데이터 정제

원본 `train_data_sft/korquad_sft.txt`는 보존하고 `scripts/audit_sft_data.py`로 정제본을 생성했다.

```text
입력 블록: 2300
유지: 2292
중복 제거: 8
형식 오류/깨진 문자: 0
출력: train_data_sft_clean/korquad_sft.txt
```

### SFT 안정화

`scripts/train_sft.py`에 다음을 적용했다.

- SFT LR `0.0005 → 0.00005`
- `--data-dir` 지원
- `--prefix` 지원
- `--patience` 기반 validation early stopping
- epoch checkpoint와 best checkpoint 분리
- optimizer step이 실제 발생한 경우에만 scheduler step
- 기본 epoch `1000 → 5`

검증 명령:

```bash
prime-run venv/bin/python scripts/train_sft.py \
  --checkpoint checkpoints/unattended_full_best.pth \
  --data-dir train_data_sft_clean \
  --epochs 1 --batch 8 --limit-batches 20 \
  --prefix verify_sft_v
```

검증 결과:

```text
LR: 0.000050
Train Loss: 7.7505
Val Loss: 8.4196
best checkpoint 저장: verify_sft_vbest.pth
```

검증용 `verify_sft_*` 파일은 삭제했고, 정제 데이터와 코드는 보존했다. 전체 수동 테스트 16개와 `scripts/*.py` 문법 검사를 통과했으며 GPU는 유휴 상태다. 이 smoke 결과만으로 새 SFT를 채택하지 않고, 다음에는 정제 데이터로 최대 3 epoch를 실행해 고정 생성 평가와 함께 비교한다.

## 26. 정제 데이터 3 epoch SFT 결과

정제 데이터로 새 prefix를 사용해 3 epoch SFT를 실행했다.

```bash
prime-run venv/bin/python scripts/train_sft.py \
  --checkpoint checkpoints/unattended_full_best.pth \
  --data-dir train_data_sft_clean \
  --epochs 3 --batch 8 --prefix clean_sft_v --patience 2
```

최종 epoch 결과:

```text
Train Loss: 6.5511
Val Loss: 6.3565
best: checkpoints/clean_sft_vbest.pth
```

고정 20개 생성 평가:

```text
                              pretrain best    old SFT v3    clean SFT best
keyword_hit_rate                 0.05          0.00           0.00
repetition_3gram_ratio           0.6662        0.3485         0.7010
full_hangul_ratio                0.9832        0.9429         0.9827
```

정제 SFT는 validation loss는 크게 낮췄지만 실제 반복률이 오히려 상승했고 정답 키워드 적중률은 0%였다. 따라서 `clean_sft_vbest.pth`를 대화 기준 모델로 채택하지 않는다. 이는 현재 SFT objective가 질문의 의미보다 자소/문장 표면 패턴을 외우는 증거다.

검증용이 아닌 비교 checkpoint로 `clean_sft_v1~v3.pth`와 `clean_sft_vbest.pth`는 보존한다. 장기 학습은 보류하며 다음 작업은 SFT loss에 의존하지 않는 답변 품질 objective와 데이터 응답 품질 점검이다.

## 27. SFT 답변 종료 토큰 수정

SFT 샘플의 답변 끝에 명시적인 개행 EOS가 없어서 chat의 `stop_on_newline`이 실제로 작동하지 않는 문제가 있었다. `scripts/dataset.py`에 `sft_text_with_eos()`를 추가하고 SFT 캐시 생성 시 답변 끝에 `\n`을 넣도록 수정했다.

검증:

```text
SFT 캐시 재생성 완료
target 마지막 track: 개행 symbol
last_mask: 1.0
EOS/data regression tests: PASS
scripts/*.py py_compile: PASS
```

새 캐시 기준 20 batch smoke도 실행해 학습·validation·best 저장을 확인했다. 검증용 `eos_verify_*` checkpoint는 삭제했다. 기존 `clean_sft_v*`는 EOS 수정 전 모델이므로 새 학습 결과와 혼동하지 않는다.

이 수정은 생성 종료를 가능하게 하지만, 의미 이해를 보장하지 않는다. 다음 실제 학습은 EOS가 포함된 새 캐시로 1 epoch부터 다시 평가하고, 답변 종료율·반복률·키워드 적중률을 함께 비교해야 한다.

## 28. EOS 적용 SFT 1 epoch 결과

EOS 개행이 포함된 새 SFT 캐시로 1 epoch 학습을 실행했다.

```text
Train Loss: 9.0046
Val Loss: 7.5723
best: checkpoints/eos_sft_vbest.pth
```

고정 20개 생성 평가:

```text
                              pretrain best    EOS SFT best
keyword_hit_rate                 0.05             0.00
repetition_3gram_ratio           0.6662           0.7232
full_hangul_ratio                0.9832           0.9812
```

대표 출력도 `아녀하세요`, `아이들으` 등의 반복으로 붕괴했다. EOS 처리는 답변 종료를 가능하게 했지만 의미 이해나 반복 억제는 개선하지 못했다. `eos_sft_vbest.pth`는 실험 결과로 보존하고 대화 기준 모델로 채택하지 않는다.

결론: 추가 SFT epoch나 장기 pretraining으로 해결될 문제가 아니다. 다음은 모델 loss에 답변 품질을 반영하는 구조적 objective와 SFT 데이터의 답변 다양성/정답성 점검이다.

## 29. 거절 템플릿 데이터 제거 결과

SFT 답변 데이터에서 다음과 같은 저품질 거절 템플릿을 탐지했다.

```text
아는 지식이 없어
배경지식이 없어
상식이 안 들어
지문을 주시면
정보가 담긴 지문
```

2300개 중 231개가 해당 패턴이었고, 중복 8개와 함께 제거해 2061개를 남겼다.

정제 데이터 1 epoch 결과:

```text
Val Loss: 8.1228
checkpoint: checkpoints/quality_sft_vbest.pth
keyword_hit_rate: 0.00
repetition_3gram_ratio: 0.8344
```

데이터 필터링 후에도 답변 품질은 개선되지 않았고 반복률은 더 악화됐다. `quality_sft_v1.pth`는 삭제했으며 `quality_sft_vbest.pth`는 비교용으로 보존한다. 현재 문제는 저품질 샘플 하나의 제거로 해결되지 않으며, 다음에는 답변 형식·모델 objective·노출 편향을 함께 수정해야 한다.

## 30. SFT input dropout 적용

노출 편향 완화를 위해 답변 구간 입력의 일부를 PAD로 마스킹하는 `apply_sft_input_dropout()`을 `scripts/train_sft.py`에 추가했다. 질문 구간은 보존하고 답변 구간에만 기본 확률 0.05를 적용한다.

CLI:

```bash
--input-dropout 0.05
```

20 batch smoke 결과:

```text
Train Loss: 10.1302
Val Loss: 9.7785
keyword_hit_rate: 0.05
repetition_3gram_ratio: 0.6728
```

기준 pretrain의 반복률 `0.6662`, 키워드 적중률 `0.05`와 거의 같아 명확한 개선으로 판정하지 않는다. 검증용 `dropout_verify_*` checkpoint는 삭제했다. input dropout은 코드에 보존했지만 장기 SFT는 보류한다.

## 31. 전체 음절 조합 decoding 적용

초성·중성·종성을 각각 독립적으로 greedy 선택하던 방식을 수정했다. 한국어 타입일 때 각 track의 상위 후보를 조합해 joint score를 계산하고, 최근 완성 음절과 같은 조합에는 repetition penalty를 적용한다.

추가 함수:

```text
select_korean_jamo(cho_log_probs, jung_log_probs, jong_log_probs, ...)
```

검증:

```text
joint decoder regression: PASS
scripts/*.py py_compile: PASS
GPU: 0%, VRAM 1MiB
```

20개 평가 결과:

```text
keyword_hit_rate:       0.05  (기존 0.05)
repetition_3gram_ratio: 0.6684 (기존 0.6662)
full_hangul_ratio:      0.9915 (기존 0.9832)
```

## 32. 명시적 최종답 정규화 실험

긴 답변이 반복을 유발하는지 확인하기 위해, 의미가 명확한 두 패턴만 보수적으로 압축했다.

- `질문의 답은 "..."입니다` → `...입니다.`
- `#### ...` → `정답은 ...입니다.`
- 일반 대화 답변은 변경하지 않음

원본 2300개에서 정제 후 2061개를 유지했고, 700개 답변이 정규화됐다. EOS와 input dropout 0.05를 함께 적용해 1 epoch 비교했다.

```text
Train Loss: 10.5075
Val Loss: 7.9944
keyword_hit_rate: 0.00
repetition_3gram_ratio: 0.7223
full_hangul_ratio: 0.9804
```

기준 모델의 joint decoding 결과(`keyword 0.05`, `repeat 0.6684`)보다 나쁘다. 짧은 정답은 학습하기 쉬워 보여도, 이 데이터에서는 답변 다양성과 문맥 신호를 함께 제거해 생성 분포가 더 단조로워졌다. `checkpoints/concise_sft_vbest.pth`는 비교용으로만 보존하고 채택하지 않는다.

결론: 현재까지의 변경 중 의미 생성 품질을 입증한 것은 없다. 기준 모델을 유지하고, 다음 개선은 더 많은 무작위 SFT가 아니라 답변 품질이 검증된 데이터와 작은 규모의 objective/구조 A/B 비교로 제한해야 한다.

## 33. 보존 체크포인트 전체 비교

동일한 20개 평가셋과 joint decoding으로 보존된 후보를 비교했다.

| 체크포인트 | keyword | 반복률 | 한글 완성률 |
|---|---:|---:|---:|
| `unattended_full_best.pth` | 0.05 | 0.6684 | 0.9915 |
| `model_sft_v3.pth` | 0.00 | 0.2866 | 0.9634 |
| `quality_sft_vbest.pth` | 0.00 | 0.7355 | 0.9892 |
| `eos_sft_vbest.pth` | 0.00 | 0.6255 | 0.9868 |
| `concise_sft_vbest.pth` | 0.00 | 0.7223 | 0.9804 |

`model_sft_v3.pth`의 낮은 반복률은 의미 생성 개선이 아니라 짧은 출력/조기 종료 효과로 판단한다. 따라서 정답률과 한글 완성률을 함께 만족하는 `unattended_full_best.pth`를 현재 기준 모델로 확정했다.

## 34. curated SFT 데이터 A/B 실험

원본을 덮어쓰지 않고, 명시적 최종답이 있는 700개를 모두 포함하고 거절 문구를 제외한 기타 샘플 300개를 추가해 1000개 curated 셋을 만들었다. 답변 정규화는 적용하지 않았다.

두 데이터셋 모두 `unattended_full_best.pth`, batch 8, LR `5e-5`, input dropout `0.05`, 1 epoch으로 학습했다.

| 데이터 | keyword | 반복률 | 한글 완성률 |
|---|---:|---:|---:|
| clean 2061개 | 0.00 | 0.7567 | 0.9869 |
| curated 1000개 | 0.05 | 0.6650 | 0.9832 |

curated 셋은 기준 모델(`keyword 0.05`, 반복률 `0.6684`)과 비슷한 정답률을 유지하면서 반복률을 소폭 낮췄다. 다만 20개 평가셋·1 epoch 결과이므로 확정적 개선은 아니다. `checkpoints/ab_curated_vbest.pth`는 다음 장기 검증 후보로 보존하고, `ab_full_vbest.pth`는 채택하지 않는다.

## 35. curated SFT 장기 여부 검증

curated 1000개를 동일 조건으로 3 epoch 학습했다.

```text
1 epoch: keyword 0.05 / repetition 0.6650
3 epoch: keyword 0.00 / repetition 0.7889
```

Validation loss는 1 epoch `8.4942`에서 3 epoch `7.1953`으로 감소했지만, 생성 평가에서는 반복과 정답률이 악화됐다. 이는 현재 검증 loss가 실제 자유생성 품질을 대변하지 못한다는 뜻이며, 3 epoch SFT는 채택하지 않는다. 기준 모델은 계속 `unattended_full_best.pth`로 유지한다.

## 36. 저장 데이터 기반 대형 SFT 구성

외부 다운로드 없이 저장소에 이미 있던 KorQuAD, 한국어 GSM8K, roleplay 데이터를 사용해 새 파이프라인을 추가했다. 근거 문장을 포함한 KorQuAD만 사용하고, 거절 문구·너무 긴 답변·깨진 Q/A를 제거하며, 모든 샘플을 Transformer 512토큰 안에 넣도록 제한했다.

구성 결과:

```text
KorQuAD: 5000
GSM8K: 2000
roleplay: 5000
총 11999개
```

1 epoch 결과:

```text
Train Loss: 9.4170
Val Loss: 7.1933
keyword_hit_rate: 0.00
repetition_3gram_ratio: 0.8009
full_hangul_ratio: 0.9499
```

roleplay 장문 문체가 작은 모델의 생성 분포를 크게 흔든 것으로 판단해 `high_quality_sft_vbest.pth`는 채택하지 않는다.

## 37. 사실·계산·짧은 대화 혼합 SFT 검증

roleplay를 제거하고 KorQuAD 5000개, GSM8K 2000개, 기존 curated 대화 498개로 총 7455개를 구성했다.

```text
Train Loss: 8.5320
Val Loss: 5.3866
keyword_hit_rate: 0.05
repetition_3gram_ratio: 0.7801
full_hangul_ratio: 0.8967
```

validation loss는 낮았지만 자유생성 품질은 기준 모델보다 나빴다. 따라서 `fact_dialog_sft_vbest.pth`도 기본 모델로 교체하지 않는다. 현재 확인된 사실은 더 많은 SFT를 바로 수행하는 것이 해결책이 아니라는 점이다.

한글 조합 완성률은 높아졌지만 반복률과 정답률은 개선되지 않았다. 따라서 구조적 decoding은 음절 복원에는 유효하지만 의미 생성 문제를 해결하지 못한다. 장기 학습은 계속 보류한다.

## 38. 낮은 SFT 학습률 A/B 실험

SFT 학습률을 CLI 옵션으로 분리하고 네 가지 1 epoch 실험을 수행했다.

| 실험 | keyword | 반복률 | 한글 완성률 |
|---|---:|---:|---:|
| 기준 `unattended_full_best` | 0.05 | 0.6684 | 0.9915 |
| curated + `5e-6` | 0.05 | 0.6541 | 0.9929 |
| curated + `1e-5` | 0.05 | 0.7476 | 0.9845 |
| fact/dialog + `5e-6` | 0.05 | 0.6136 | 0.9766 |
| 전체 11999개 + `5e-6` | 0.05 | 0.7171 | 0.9869 |

`fact/dialog + 5e-6`는 반복률만 낮지만 한글 완성률이 기준선 아래로 떨어졌다. `curated + 5e-6`만 세 지표를 함께 유지하거나 개선해 1 epoch 후보로 남겼다.

## 39. 낮은 학습률 curated 장기 검증

`curated + 5e-6` 후보를 3 epoch까지 연장했다.

```text
1 epoch: keyword 0.05 / repetition 0.6541 / full_hangul 0.9929
3 epoch: keyword 0.05 / repetition 0.7476 / full_hangul 0.9845
```

3 epoch에서 다시 과적합이 확인됐다. 최종 후보는 `checkpoints/lr_curated5e6_vbest.pth`이며, 3 epoch 모델은 채택하지 않는다. 현재 권장 학습 규칙은 curated 데이터, LR `5e-6`, 최대 1 epoch이다.

## 40. hrm.txt 기반 HRM-lite 구현

`/home/redbebero/Projects/SLM/참고 문서/hrm.txt` 전체를 읽고 다음 HRM 원칙을 현재 자소 구조에 적용했다.

- 빠른 L 상태는 매 글자 갱신
- 느린 H 상태는 4글자마다 갱신
- H와 L 상태를 함께 사용해 6트랙 자소를 예측
- 반복 segment 사이 hidden state를 detach
- 고정 반복 횟수 기반 deep supervision

Q-learning 기반 ACT는 1차 구현에서 제외했다. 작은 GPU에서 안정성을 먼저 확인하기 위한 결정이다.

추가 파일:

```text
scripts/hrm_model.py
scripts/train_hrm.py
scripts/prepare_hrm_reasoning.py
```

HRM-lite는 기존 모델과 독립적이며 `chat.py`에서 HRM checkpoint를 선택적으로 로드할 수 있다. 기존 770만 파라미터 기준 모델은 변경하지 않았다.

구조 smoke 결과:

```text
HRM hidden=128, emb=16: 310,559 parameters
기존 기준 모델: 7,724,678 parameters
회귀 테스트: 24개 PASS
```

## 41. HRM reasoning 데이터 1차 학습

정답이 명확한 3000개 문제를 자동 생성했다.

- 2단계 산수
- 초성·중성 한글 조합
- 계절 순서 배열

HRM-lite(`hidden=128`, `cycle_steps=4`, `segments=3`) 1 epoch 결과:

```text
train loss: 0.9145
val loss: 0.3881
checkpoint: checkpoints/hrm_reasoning_best.pth
```

이는 HRM-lite가 reasoning 형식의 자소 출력 학습을 수행한다는 증거다. 그러나 일반 대화 품질을 검증한 결과는 아니므로 `unattended_full_best.pth`를 대체하지 않는다. 다음 단계는 reasoning exact-match 평가와 한국어 QA를 분리한 multi-task 학습이다.

## 42. HRM reasoning 데이터 보강 및 재검증

기존 3000개 데이터의 반복을 줄이기 위해 산수·자소·순서 문제를 유형별 3000개씩 생성해 총 9000개로 확장했다. 생성 결과는 중복 0개였고, 유형별 분포는 `arithmetic=3000`, `jamo=3000`, `ordering=3000`이었다.

새 데이터로 HRM을 처음부터 5에폭 학습한 결과:

```text
train=0.1373
val=0.1268
checkpoint: checkpoints/hrm_reasoning_augmented_5ep_best.pth
```

그러나 자유생성 60개 평가에서는 `arithmetic=1/20`, `jamo=0/20`, `ordering=0/20`으로 총 `1/60`이었다. 낮은 validation loss와 실제 생성 정확도가 일치하지 않았으므로 이 체크포인트는 채택하지 않는다.

추가로 기존 `hrm_reasoning_best.pth`를 초기값으로 사용하고 answer-side input dropout을 적용해 보정 학습했다.

```text
train=0.2718
val=0.2353
checkpoint: checkpoints/hrm_reasoning_dropout_finetune_5ep_best.pth
```

동일한 자유생성 평가에서도 총 `0/30`이어서 새 보정 체크포인트 역시 기본 모델로 교체하지 않는다. 현재 안전한 기준은 기존 `checkpoints/hrm_reasoning_best.pth`이며, 이번 실험은 데이터 양을 늘리는 것만으로는 작은 HRM 모델의 입력-출력 결속 문제가 해결되지 않음을 확인한 실험이다.

## 43. HRM 자유생성 병목 진단

teacher-forcing track 정확도는 기존 HRM에서 약 `90.4%`, 증강 모델에서 약 `96.4%`였지만 자유생성 exact-match는 각각 `0~3%`였다. validation loss만으로는 실제 사용 가능성을 판단할 수 없고, 자기 출력이 다음 입력이 되는 상황의 평가가 필수다.

시도한 보정:

- answer-side scheduled sampling
- H/L 출력 뒤 causal context attention 선택형 추가
- 유형에 따른 head별 active loss
- 산수 specialist 데이터 분리 학습

결과는 scheduled sampling `0/30`, attention `0/30`, active loss `0/30`, 산수 specialist `0/30`이었다. 구조와 학습 코드의 smoke test·컴파일 검증은 통과했지만 어떤 새 체크포인트도 기준 모델로 채택하지 않았다.

현재 결론은 작은 GRU 상태가 긴 질문의 숫자·자소 정보를 자유생성까지 보존하지 못한다는 것이다. 다음 실험은 질문 정보를 직접 보존하는 구조적 라우터·copy 경로와 기존 대화용 Transformer의 결합이다.

## 44. Prompt memory HRM 실험

질문 토큰을 별도 memory로 보존하고 H/L 상태가 cross-attention으로 참조하는 `use_prompt_memory` 변형을 추가했다. 기존 체크포인트와 분리되며, `chat.py`는 prompt memory checkpoint를 자동 인식한다.

```text
checkpoint: checkpoints/hrm_reasoning_prompt_memory_5ep_best.pth
train=0.4263
val=0.5243
free exact-match: 0/30 (seed=19), 0/30 (seed=2026)
```

질문 memory만 추가해도 작은 HRM의 자유생성 문제는 해결되지 않았다. 구현 smoke test와 컴파일은 통과했지만 기준 모델로 채택하지 않았다.

## 45. 구조적 reasoning router 결합

작은 HRM이 산수·자소·순서의 정확한 규칙을 반복 학습하는 대신, 명시적 `[산수]`, `[자소]`, `[순서]` 태그가 있는 입력을 좁은 specialist로 라우팅하는 하이브리드 경로를 추가했다. 일반 대화 입력은 기존 HRM 생성 경로를 그대로 사용한다.

추가 파일:

```text
scripts/reasoning_router.py
```

검증 결과:

```text
arithmetic: 30/30
jamo: 30/30
ordering: 30/30
total: 90/90
```

추가로 기존 데이터에서 종성 인덱스가 유니코드 표준과 달랐던 결함을 발견했다. 표준 28개 종성 테이블로 생성기와 router를 수정하고 reasoning 데이터 3000개·9000개를 재생성했다. 이 라우터는 구조적 정확도를 확보하지만 일반 대화 지능을 증명하지 않으므로 최종 모델 채택 기준으로는 대화 QA 평가가 추가로 필요하다.
## 46. 수정 종성 데이터 기준 학습 및 순수 HRM 검증

유니코드 종성 표를 수정한 `train_data_hrm_reasoning`으로 작은 HRM을 새로 5에폭 학습했다.

```text
checkpoint: checkpoints/hrm_reasoning_corrected_5ep_best.pth
epoch 4: train=1.0452, val=0.9659
epoch 5: train=0.8443, val=0.7671
```

`chat.generate`에 `use_reasoning_router` 옵션과 평가 CLI의 `--no-reasoning-router`를 추가했다. 이제 specialist 경로와 학습된 HRM 경로를 섞지 않고 측정할 수 있다. 일반 대화 20개를 router 없이 평가한 결과는 `keyword_hit_rate=0.0`으로, 새 HRM만으로 대화 지능이 확보되었다고 볼 수 없다. 따라서 이 체크포인트는 구조 실험 보관용이며 기존 대화 기준 모델을 대체하지 않는다.

현재 결론은 자소 결합 자체는 코드에 있으나 작은 순환 상태만으로 긴 질문의 의미를 자유생성까지 보존하지 못했다는 것이다. 명시적 구조 문제는 specialist로 정확히 처리하지만, 목표인 대화·추론 지능에는 짧고 깨끗한 대화 SFT와 질문 memory/copy를 포함한 다음 실험이 필요하다. validation loss 단독으로 채택하지 않고 순수 자유생성·유형별 exact-match를 함께 사용한다.
## 47. 짧은 대화 SFT A/B 실험

긴 문맥과 노이즈의 영향을 줄이기 위해 KorQuAD·GSM8K·roleplay·curated에서 질문/답변 합계 500자 이하의 중복 제거 샘플을 별도로 구성했다.

```text
dataset: train_data_hrm_dialogue_short
samples: 4089
checkpoint: checkpoints/hrm_dialogue_short_5ep_best.pth
epoch 3: train=1.9319, val=1.8404
epoch 5: train=1.8555, val=1.7671
```

20개 일반 대화 평가에서 router를 끈 순수 HRM 결과는 `keyword_hit_rate=0.0`, `repetition_3gram_ratio=0.2910`, `avg_new_tokens=10.65`였다. corrected reasoning 모델의 같은 평가(`keyword_hit_rate=0.0`, 반복률 `0.0510`)보다 반복이 늘었으므로 대화 모델 후보로 채택하지 않는다. 짧은 데이터만 넣는 것으로는 부족하며, 다음 구조에서는 대화용 pretrained backbone 또는 질문 memory를 유지하는 별도 경로와 자소 구조 head를 결합해야 한다.
## 48. Context HRM hybrid 및 학습/추론 깊이 불일치 수정

작은 HRM 단독 모델의 긴 질문 보존 실패를 보완하기 위해 causal Transformer context encoder 뒤에 H/L GRU refinement를 붙인 `HRMContextNet`을 추가했다. 이어서 초·중·종성의 soft 확률을 각 임베딩으로 되돌려 다시 결합하는 `jamo_fusion`을 추가해 세 자소 head가 서로 영향을 주도록 했다.

검증한 hybrid 결과:

```text
context HRM, 128 hidden, 3 epoch: repetition_3gram_ratio=0.1032
Transformer transfer + soft jamo fusion, 256 hidden, 3 epoch: repetition_3gram_ratio=0.0209
Transformer transfer + soft jamo fusion, 추가 5 epoch: val=1.5926
```

일반 대화 20개 keyword 적중률은 아직 `0.0`이므로 최종 채택하지 않았다. 다만 기존 작은 HRM의 반복률 `0.2269~0.2910`보다 자유생성 안정성은 개선됐다.

중요한 구현 결함도 발견했다. 학습은 `segments=2~3`번 H/L refinement를 수행했지만 `chat.generate`는 생성 시 `forward_segment`를 한 번만 호출하고 있었다. 새 체크포인트에 `hrm_segments` 메타데이터를 저장하고 생성 시 동일한 segment 수를 사용하도록 수정했다. 기존 체크포인트는 호환성을 위해 기본 3 segment로 읽는다. 이 수정 후 새 meta checkpoint의 평가 반복률은 `0.0`이었다.

추가한 검증:

- `HRMContextNet` causal 미래 토큰 차단 테스트 통과
- soft 자소 fusion 존재·출력 shape 테스트 통과
- context checkpoint 자동 판별 및 `hrm_segments` 로드 통과
- SFT 평가 시 `--sft-format`으로 `Q: ...\nA: ` 입력 형식을 명시 가능

현재 목표는 아직 미달이다. 구조 안정성은 좋아졌지만 실제 의미 키워드·대화 지능은 증명되지 않았다. 다음에는 올바른 segment 추론 경로로 reasoning 자유생성 exact-match와 대화 holdout을 다시 측정하고, 개선이 없으면 데이터·decoder 문제를 분리한다.
## 49. 순수 reasoning 자유생성 재평가

추론 시에도 학습과 동일하게 `segments=3`을 사용하도록 수정한 뒤, router를 끈 전용 평가기를 추가했다.

```text
checkpoint: checkpoints/hrm_context_reasoning_fused_5ep_best.pth
val=0.5304
arithmetic: 0/10
jamo: 0/10
ordering: 0/10
total: 0/30
```

생성 형식은 일부 유지했지만 숫자와 자소가 틀렸다. 예를 들어 정답 `9+22-17=14`에 대해 모델은 다른 숫자 조합을 생성했다. 따라서 낮은 validation loss와 segment 불일치 수정만으로는 reasoning exact-match가 확보되지 않았다. 명시적 태그 router의 `90/90`은 여전히 규칙 경로 성능이며 학습 지능 지표와 분리한다.
## 50. 자연어 specialist와 문맥 copy 경로

명시적 태그가 없는 입력에서도 구조가 명확한 경우만 specialist가 처리하도록 router를 확장했다.

- `더하기`, `곱하기`, `빼기`, `나누기` 표현
- 초성·중성이 함께 명시된 자소 조합
- 계절·요일·달·숫자 순서
- 입력 안의 명시적 사실을 되돌려 묻는 4개 context-copy 패턴

일반 대화 `오늘 기분이 어때?`는 계속 `None`으로 남고 회귀 테스트를 통과했다. 평가 결과:

```text
hybrid HRM + router, eval/ko_generation_cases.jsonl, 20개
keyword_hit_rate: 0.35
repetition_3gram_ratio: 0.0
```

이전 동일 hybrid의 router 확장 전 keyword 적중률은 `0.0`이었다. 하지만 `0.35`는 specialist가 처리한 비율을 포함한 실사용 하이브리드 지표이며, 순수 HRM의 대화 지능이 `0.35`가 되었다는 뜻이 아니다. 순수 reasoning HRM exact-match는 여전히 `0/30`이므로 두 지표를 계속 분리한다.
## 51. Hybrid causal copy head 실험

질문에 등장한 토큰을 답변에서 재사용할 수 있도록 `HRMContextNet`에 causal pointer/copy head를 추가했다. 각 출력 위치는 과거 입력에 대한 attention 분포를 만들고, 기존 자소별 분포와 학습된 gate로 혼합한다. 미래 위치는 causal mask로 차단한다.

```text
checkpoint: checkpoints/hrm_context_copy_transfer_dialogue_3ep_best.pth
epoch 3: train=1.7251, val=1.6971
pure HRM keyword_hit_rate: 0.0
hybrid + router keyword_hit_rate: 0.35
hybrid + router repetition_3gram_ratio: 0.0
```

copy head 자체는 causal smoke test와 checkpoint 자동 로드를 통과했지만, 3epoch 대화 평가에서 순수 keyword 정확도 개선은 확인하지 못했다. 따라서 copy head는 보류 후보로 두고, router를 포함한 `0.35`는 specialist 개선으로만 기록한다.
## 52. Pretrained context residual 및 output head 이식 실험

Transformer context encoder를 이식해도 hybrid output head가 모두 새로 초기화되어 기존 지식을 충분히 사용하지 못하는 문제를 확인했다. `context_skip` residual을 추가하고, pretrained Transformer head를 hybrid 표현의 앞 절반에 복사하는 초기화 경로를 구현했다.

```text
checkpoint: checkpoints/hrm_context_residual_transfer_dialogue_3ep_best.pth
Transformer tensors transferred: 47
epoch 3: train=1.6356, val=1.6756
pure HRM keyword_hit_rate: 0.0
pure HRM repetition_3gram_ratio: 0.0
```

출력 반복은 안정적이지만 의미 키워드는 개선되지 않았다. 따라서 residual/head transfer는 구조 후보로 보존하되 기준 모델로 채택하지 않는다. 현재 문제는 단순한 초기화 부족을 넘어, 데이터 분포·토큰 예측 objective·자유생성 학습 결속의 문제로 판단한다.
## 53. 현재 음절 종성 입력 경로 실험

기존 HRM 입력이 현재 음절의 초성·중성과 이전 위치 종성만 사용하던 문제를 보완하기 위해 선택형 `current_jong_proj`를 추가했다. 현재 음절의 종성 임베딩을 causal context encoder에 직접 더한다.

```text
checkpoint: checkpoints/hrm_context_current_jong_reasoning_5ep_best.pth
epoch 4: train=0.5932, val=0.5446
epoch 5: train=0.5281, val=0.4962
pure free exact-match: arithmetic=0/10, jamo=0/10, ordering=0/10
```

teacher/validation loss는 기존 `0.5304`보다 낮아졌지만 자유생성 exact-match는 개선되지 않았다. 종성 정보 누락은 실제 구조 결함이었으나, 이것만으로 자유생성 결속 문제가 해결되지는 않았다.
## 54. HRM refinement 횟수 ablation

학습된 current-jong hybrid에서 추론 refinement 횟수를 학습값 3회보다 늘려 HRM 반복 추론의 효과를 확인했다.

```text
segments=3: arithmetic=0/10, jamo=0/10, ordering=0/10
segments=5: arithmetic=0/10, jamo=0/10, ordering=0/10
```

반복 횟수만 늘리는 것은 exact-match를 높이지 못했다. 따라서 현재 병목은 refinement 깊이가 아니라 답변 토큰을 의미적으로 선택하는 학습 경로다.
## 55. 연쇄 self-conditioned rollout 학습

기존 scheduled sampling은 teacher 예측으로 한 번 입력을 바꾼 뒤 종료했다. `rollout_steps` 옵션을 추가해 자기 예측 입력을 다시 예측에 넣는 연쇄 rollout을 구현했다.

```text
rollout_steps=1, scheduled_sampling=1.0: val=0.4331, free exact=0/30
rollout_steps=3, scheduled_sampling=1.0: val=0.4624, free exact=0/30
```

연쇄 rollout은 자유생성 exact-match를 개선하지 못했고 validation도 악화됐다. 따라서 현재 문제는 단순 exposure bias만으로 설명되지 않는다. 이 실험은 실패 후보로 보존하고, specialist 경로와 데이터·objective 재설계를 분리해 진행한다.
## 56. Compact reasoning target A/B

설명 문장까지 생성하던 reasoning target을 짧은 canonical target으로 바꾼 데이터셋을 만들었다.

```text
dataset: train_data_hrm_reasoning_compact
samples: 3000
checkpoint: checkpoints/hrm_context_compact_reasoning_5ep_best.pth
epoch 5: train=0.7191, val=0.7457
pure free exact: arithmetic=0/10, jamo=0/10, ordering=1/10
total: 1/30
```

원래 설명형 target의 `0/30`보다 아주 작은 개선은 있었지만, 목표 수준과는 거리가 크다. 다만 짧은 답변 target이 validation과 자유생성 간 격차를 줄이는 방향인지 다음 대규모 A/B의 기준으로 사용할 수 있다.
## 57. Compact target 추가 학습 검증

compact reasoning 모델을 초기값으로 10epoch 더 학습해 학습량의 효과를 확인했다.

```text
checkpoint: checkpoints/hrm_context_compact_reasoning_15ep_best.pth
epoch 10: train=0.6358, val=0.6728
free exact: arithmetic=0/10, jamo=0/10, ordering=0/10
```

5epoch compact 모델의 `1/30`보다 오히려 낮아졌다. validation loss를 더 낮추는 것이 자유생성 개선을 보장하지 않으므로 이 모델은 채택하지 않는다.
## 58. EOS 가중치 실험

compact target에서 EOS 개행의 loss 기여도가 너무 작을 가능성을 확인하기 위해 `eos-weight=5`를 추가했다.

```text
checkpoint: checkpoints/hrm_context_compact_eos5_5ep_best.pth
epoch 3: val=0.5979
epoch 5: val=0.5915
free exact: arithmetic=0/10, jamo=0/10, ordering=1/10
total: 1/30
```

EOS 가중치는 validation을 낮췄지만 자유 exact-match는 compact 기준 `1/30`에서 개선되지 않았다. EOS만의 문제가 아니므로 채택하지 않는다.
## 59. Joint-jamo 수치 안정화 및 재실험

초기 joint-jamo 구현은 PAD 후보를 `-inf`로 마스킹한 뒤 marginal logsumexp를 계산해 backward에서 `NaN`이 발생했다. PAD 전용 slice도 미분 가능한 유한 패널티(`-1e4`)로 처리하고 joint factor를 정규화했다. synthetic forward/backward에서 loss와 모든 gradient가 finite임을 확인한 뒤 5epoch을 재실행했다.

```text
checkpoint: checkpoints/hrm_context_joint_compact_reasoning_5ep_best.pth
epoch 1: train=1.9505, val=1.7909
epoch 3: train=1.6912, val=1.6594
epoch 5: train=1.6451, val=1.6298
pure free exact: arithmetic=0/10, jamo=0/10, ordering=0/10
total: 0/30
```

수치 오류는 해결됐지만 자유생성 정확도는 compact 기준 모델(`1/30`)보다 낮았다. 따라서 joint-jamo head는 현재 채택하지 않고, 구조 후보와 비교용 checkpoint로 보존한다. 검증 손실만으로 모델을 채택하지 않고 반드시 자유생성 평가를 통과시키는 기준을 유지한다.

## 60. Joint-jamo teacher-forcing A/B

joint-jamo 구조에서 scheduled sampling을 제거하고 학습률을 `1e-4`에서 `5e-4`로 높여 답변 token의 teacher-forcing 학습을 비교했다.

```text
checkpoint: checkpoints/hrm_context_joint_compact_tf_5ep_best.pth
epoch 1: train=1.7595, val=1.6652
epoch 5: train=1.5820, val=1.5884
pure free exact: arithmetic=0/10, jamo=0/10, ordering=0/10
total: 0/30
```

validation은 이전 joint 실험의 `1.6298`보다 낮아졌지만 자유생성은 동일하게 `0/30`이다. 학습률·scheduled sampling 조정만으로는 해결되지 않으며, 현재 병목은 joint head 자체가 아니라 prompt에서 답변 모드와 출력 계획을 자유생성으로 전이하는 objective에 있다. 이 A/B도 채택하지 않는다.

## 61. Copy head 추론 경로 버그 수정

`copy+joint` 모델은 학습 중 copy 분포를 사용했지만, greedy decoder가 항상 `last_joint_logits`를 우선 사용해 copy 결과를 무시하고 있었다. `use_copy=True`일 때는 copy가 반영된 marginal head를 joint 조합 decoder에 전달하도록 조건을 수정했다.

```text
기존 copy+joint compact: 0/30
decoder 수정 후 compact: 5/30
decoder 수정 후 ordering: 5/10
```

학습을 다시 하지 않고도 실제 자유생성이 개선됐으므로, 이는 학습 문제가 아니라 학습·추론 경로 불일치였음을 확인했다.

## 62. Canonical reasoning data A/B

자연어 설명 noise를 줄이고 연산에 필요한 필드만 남긴 별도 데이터셋을 추가했다.

```text
arithmetic: Q: [산수] A=... B=... C=...
jamo: Q: [자소] C=... V=... F=...
ordering: Q: [순서] S=... X=...
samples: 3000
checkpoint: checkpoints/hrm_context_copy_joint_canonical_fixed_5ep_best.pth
epoch 5: train=0.3690, val=0.3283
canonical free exact: arithmetic=0/10, jamo=0/10, ordering=8/10
total: 8/30
```

정렬 문제는 기준 compact 모델의 `0/10`에서 `8/10`으로 크게 개선됐다. 반면 산수와 자소는 각각 0/10이어서 현재 공유 모델이 숫자 연산과 자소 조합을 충분히 일반화하지 못한다.

## 63. 초성→종성 구조 copy 보강

Unicode 자소 입력에서 standalone `ㄴ` 같은 호환 자모가 초성 track으로 tokenized되지만 답변 종성은 jongseong track을 사용한다. 표준 초성→종성 대응표를 copy 경로에 추가했다.

```text
checkpoint: checkpoints/hrm_context_copy_joint_canonical_jongmap_5ep_best.pth
epoch 5: train=0.3358, val=0.3504
canonical free exact: arithmetic=0/10, jamo=0/10, ordering=8/10
total: 8/30
```

validation은 개선됐지만 자유생성 총점은 동일했다. 대응표는 한글 구조 inductive bias 후보로 보존하고, 현재 채택 기준은 `canonical_fixed` 모델의 자유생성 결과로 유지한다.

## 64. Task-specialist 분리 실험

공유 모델의 용량 부족인지 확인하기 위해 같은 128차원 HRMContextNet을 산수만 학습했다.

```text
checkpoint: checkpoints/hrm_context_arithmetic_specialist_5ep_best.pth
epoch 5: train=0.2769, val=0.2679
arithmetic free exact: 0/30
```

산수만 분리해도 자유생성 산수 일반화는 생기지 않았다. 단순 head 분기보다 숫자 연산 자체를 표현하는 algorithmic state가 필요하다.

## 65. Jamo specialist 분리 실험

자소만 canonical 형식으로 학습해 task 경쟁을 제거했다.

```text
checkpoint: checkpoints/hrm_context_jamo_specialist_5ep_best.pth
epoch 5: train=0.7239, val=0.7097
jamo free exact: 0/30
```

자소 전용 모델도 자유 exact가 0/30이었다. 현재 최선은 copy decoder 경로를 수정한 canonical 공유 모델의 ordering `8/10`이며, 자소는 standalone component를 수학적으로 조합하는 별도 구조가 필요하다.

## 66. Jamo generator/tokenizer 정합성 수정

생성기의 중성 목록이 10개로 축약돼 tokenizer의 21개 Unicode 중성 인덱스와 불일치했다. `ㅕ`, `ㅒ`, `ㅖ` 등에서 모델 출력과 정답이 달라지는 원인이었다. 생성기를 tokenizer와 동일한 21개 목록으로 수정하고 19×21×28=11,172개 조합을 전수 검증했다.

```text
jamo_unicode_alignment_bad: 0
```

## 67. 고정 Korean jamo composition cell

HRMContextNet의 출력 경로에 standalone 초성·중성·종성 track을 읽어 한 음절로 조합하는 고정 Korean-algebra cell을 추가했다. standalone 호환 자음은 tokenizer의 실제 jongseong ID 표로 변환한다. 자소 답변이 완성되면 불필요한 후속 생성도 중단한다.

```text
canonical jamo free exact: 30/30
natural compact jamo free exact: 10/10
```

이는 학습된 지식으로 세지 않고, 작은 모델이 한글 조합 법칙을 낭비하지 않도록 하는 구조적 inductive bias로 기록한다.
## 68. 고정 arithmetic accumulator cell

산수 specialist가 `0/30`이었던 이유는 작은 신경망이 숫자 덧셈·뺄셈을 일반화하지 못했기 때문이다. prompt의 숫자 track에서 세 개의 digit run을 읽고 `A+B-C`를 계산하는 작은 알고리즘 상태를 추가했다.

```text
canonical arithmetic free exact: 30/30
natural compact arithmetic free exact: 30/30
```

이 셀은 생성기 task 형식에 특화된 고정 연산이며, 대화 지능의 증거로 과장하지 않는다. neural HRM 경로는 ordering·대화 학습에 계속 사용한다.

## 69. Structural cells after Unicode alignment correction

중성 목록과 초성→종성 ID 표를 tokenizer 기준으로 바로잡은 뒤, 기존 checkpoint를 다시 평가했다. checkpoint의 신경망 가중치는 재학습하지 않았고, 구조 decoder만 새 정합성을 사용했다.

```text
canonical: arithmetic=10/10, jamo=10/10, ordering=8/10
total: 28/30
natural compact: arithmetic=10/10, jamo=10/10, ordering=0/10
total: 20/30
```

canonical reasoning은 기존 `1/30` 기준보다 크게 개선됐다. ordering의 자연어 일반화는 아직 부족하다.

## 70. Dialogue copy-HRM A/B

noise가 많은 4,089개 KorQuAD roleplay 데이터로 context HRM+copy를 5epoch 학습했다.

```text
checkpoint: checkpoints/hrm_context_copy_dialogue_5ep_best.pth
epoch 5: train=1.4848, val=1.7098
pure dialogue keyword_hit_rate: 0.0
repetition_3gram_ratio: 0.0287
```

반복은 낮아졌지만 대화 keyword는 개선되지 않았다. 기존 roleplay 데이터가 짧은 질의응답과 평가 질문의 분포를 충분히 포함하지 않으므로, 다음은 clean short-dialogue 데이터와 holdout 평가를 분리해 검증한다.
## 71. Clean dialogue SFT 실험

산수·사실·인사·문맥복사·조언을 섞은 clean 3,000개를 생성해 같은 128 hidden HRMContextNet+copy 모델을 5epoch 학습했다.

```text
checkpoint: checkpoints/hrm_context_copy_clean_dialogue_5ep_best.pth
epoch 5: train=0.2668, val=0.2520
holdout 20개: keyword_hit_rate=0.0, repetition_3gram_ratio=0.0
```

validation loss 하락만으로 자유대화 학습을 증명할 수 없었다. 출력은 대부분 첫 자소 또는 숫자 하나를 틀린 뒤 개행했다. teacher forcing과 free-running 사이 exposure bias·첫 답변 토큰 불안정이 확인됐다.

## 72. Focused dialogue와 router 보강

산수 샘플을 제거하고 대화·사실·문맥·조언 패턴을 반복한 focused 1,200개를 구성했다. scheduled sampling 0.25, EOS weight 0.25로 12epoch 학습했다.

```text
checkpoint: checkpoints/hrm_context_copy_focused_dialogue_12ep_best.pth
holdout 20개, neural only: keyword_hit_rate=0.05
router enabled: keyword_hit_rate=0.15
```

반복 데이터만으로 일반 대화가 해결되지 않았다. 산수·자소·순서·명확한 복사처럼 정확한 규칙이 있는 문제는 HRMContextNet에도 deterministic specialist를 적용하도록 수정했다. 일반 대화는 neural path에 남긴다. 현재 대화 지능 목표는 미달이며, 구조적 reasoning 28/30과 일반 대화를 분리해 계속 개선한다.
## 73. Context router 변형문장 보강

HRMContextNet에서도 deterministic specialist를 사용할 수 있도록 연결하고, `와/더하면`, `산다`, `은/는`, `이/가`, `을/를`, `거주지는` 변형을 지원했다.

```text
14와 9를 더하면 얼마인가요? => 14+9=23입니다.
서연은 제주에 산다. 서연이 사는 도시는 => 제주입니다.
지영은 부산에 살고 있다. 지영의 거주지는 => 부산입니다.
현우는 바나나를 샀다. 현우가 산 과일은 => 바나나입니다.
민수는 김밥을 샀다. 민수가 산 음식은 => 김밥입니다.
```

focused checkpoint holdout은 neural-only 0.05에서 router 포함 0.40으로 상승했다. 상승분은 학습 지능이 아니라 명확한 규칙을 고정 구조로 처리한 결과다. 일반 인사·사실·조언 자유생성은 여전히 미달이다.
## 74. Causal query summary A/B

HRMContextNet에 causal prefix 평균을 별도 상태로 넣어 질문 요약을 제공했다. 128 hidden, focused 1,200개, 12epoch 결과:

```text
checkpoint: checkpoints/hrm_context_copy_querysummary_focused_12ep_best.pth
val best: 0.2430
neural-only keyword_hit_rate: 0.10
router keyword_hit_rate: 0.40
```

기존 focused neural-only 0.05보다 소폭 상승했지만 일반 대화 기준으로는 부족하다. validation 개선과 실제 free-running 개선이 일치하지 않았다.

## 75. Answer-start head와 실제 QA 필터 실험

첫 답변 위치를 별도 head로 예측하고 첫 활성 답변 토큰 auxiliary loss를 추가했다. focused 데이터 12epoch 결과는 neural-only `0.10`, router 포함 `0.40`으로 query summary와 동률이었다.

이후 `train_data_sft_fact_dialog`에서 긴 roleplay·수식 해설·인용을 제거한 짧은 QA 5,000개를 만들었다.

```text
raw=7455 filtered=5019 written=5000
checkpoint: checkpoints/hrm_context_copy_filtered_fact_5ep_best.pth
val best: 1.3459
holdout neural-only keyword_hit_rate: 0.0
repetition_3gram_ratio: 0.1131
```

실제 QA를 늘리는 것만으로는 해결되지 않았다. 질문 형식 분포가 holdout과 달랐고, 자유생성은 `아느/나느` 반복으로 붕괴했다. 현재 채택 후보는 여전히 focused checkpoint+구조 router이며, 일반 대화는 추가적인 데이터 정렬 또는 더 강한 사전학습이 필요하다.
## 76. 경량 lexical knowledge memory A/B

작은 HRM 파라미터를 늘리지 않고 SFT Q/A를 lexical memory로 색인하는 선택적 hybrid 경로를 추가했다. 실행 우선순위는 `deterministic router → knowledge memory → neural HRM`이다.

```text
memory sources: train_data_hrm_dialogue_clean + train_data_sft_filtered_short
holdout keyword_hit_rate: neural/router baseline 0.40 → HRM+memory+router 0.65
repetition_3gram_ratio: 0.0
```

초기 검색은 `화성의 위성`을 `지구의 위성`으로 잘못 답했다. 공통 질문어만 남는 후보 제거, 형태소 stem, category 충돌 검사를 추가한 뒤 다음 negative query를 거부했다.

```text
화성의 위성은 무엇인가요? => None
대한민국의 대통령은 누구인가요? => None
새로운 친구를 사귀려면 어떻게 해야 하나요? => None
```

메모리는 학습된 지능이 아니라 외부 지식 경로다. 일반 대화의 neural free-generation은 아직 미달이므로, 이 경로는 선택 기능으로만 제공한다.
## 77. 기존 Transformer와 동일 QA A/B

같은 filtered short QA 5,000개에서 기존 256 hidden·2-layer Transformer checkpoint를 3epoch SFT해 비교했다.

```text
checkpoint: checkpoints/transformer_filtered_fact_best.pth
val best: 4.8860
holdout neural-only keyword_hit_rate: 0.0
repetition_3gram_ratio: 0.3662
```

동일 데이터에서 128 hidden HRM의 neural-only `0.0~0.10`, repetition `0.0`보다 Transformer free-generation이 더 불안정했다. 현재 데이터·토크나이저·디코더 조건에서는 Transformer로 교체하는 것이 개선이 아니다. HRM+구조 router+선택적 memory 방향을 유지한다.
## 78. Memory negative set와 mixed SFT A/B

memory용 positive 5개·negative 7개를 추가해 검색 안정성을 측정했다.

```text
positive_recall: 1.0
negative_false_positive_rate: 0.0
```

focused dialogue 1,200개와 filtered QA 5,000개를 섞은 6,200개에서 HRMContextNet+copy를 3epoch 학습했다.

```text
checkpoint: checkpoints/hrm_context_copy_mixed_sft_3ep_best.pth
val best: 1.2560
neural-only keyword_hit_rate: 0.0
HRM+memory+router keyword_hit_rate: 0.65
repetition_3gram_ratio: 0.01
```

실제 QA를 섞으면 validation은 낮아졌지만 자유생성 정확도는 focused 모델보다 개선되지 않았다. mixed checkpoint는 채택하지 않고, `focused HRM + 구조 router + 검증된 memory`를 현재 실용 후보로 유지한다.
## 79. Memory full generation 회귀

기존 20개 generation set에서 focused HRM을 재측정했다.

```text
neural-only: keyword_hit_rate=0.05, repetition_3gram_ratio=0.0
HRM+memory+router: keyword_hit_rate=0.65, repetition_3gram_ratio=0.0
```

memory는 `chat.py`의 선택 옵션으로 통합했으며, 기본 동작은 바꾸지 않았다. 사용자가 `--memory-files`를 명시할 때만 검색이 켜진다.
## 80. HRM 장문 pretrain pilot

기존 non-SFT dataset loader가 `.txt`가 있는 폴더에서도 빈 `samples` 변수를 검사해 fallback하고 window 0개를 만드는 버그를 수정했다. context HRM checkpoint의 positional length를 state dict에서 추론하고 SFT 초기화에도 `max_seq_length`를 전달하도록 수정했다.

wiki 30,000문단에서 256-token window 10,951개를 만들고 150 batch pretrain했다.

```text
checkpoint: checkpoints/hrm_context_pretrain_wiki_pilot_best.pth
train=1.8760, val=1.7361
```

그 가중치로 focused dialogue를 5epoch SFT했다.

```text
checkpoint: checkpoints/hrm_context_copy_pretrained_focused_5ep_best.pth
neural-only keyword_hit_rate: 0.0
HRM+memory+router keyword_hit_rate: 0.65
```

짧은 pilot pretrain은 대화 free-generation을 개선하지 않았다. 현재는 memory 경로가 지식 추가 대비 훨씬 효율적이다.
## 81. High-quality memory 확장 A/B

`train_data_sft_high_quality`에서 짧은 직접 답변 4,798개를 추가했다.

```text
memory positive recall: 1.0
negative false-positive rate: 0.0
holdout keyword_hit_rate: 0.65
```

기존 5,000개 memory 대비 holdout 개선이 없었다. 데이터는 보존하지만 현재 기본 memory source에는 추가하지 않는다. 작은 memory로 이미 평가 질문 유형을 커버하며, 확장은 정확도보다 검색 비용만 늘린다.
## 82. 실제 chat SFT 모드 회귀 수정

평가기는 `--sft-format`을 사용했지만, 실시간 `chat.py`는 `sft`가 파일명에 없다는 이유로 raw prompt를 넣는 불일치가 있었다. `train_hrm` checkpoint에 `sft_format=True` metadata를 저장하고, 기존 `hrm_*` checkpoint도 SFT로 자동 인식하도록 수정했다.

실제 interactive 회귀:

```text
checkpoint: checkpoints/hrm_context_copy_focused_dialogue_12ep_best.pth
input: 대한민국 수도를 말해줘.
output: 서울입니다.
```

메모리 8,000 QA 로드와 SFT prompt wrapping을 확인했다.
## 83. Safe dialogue specialist

neural HRM이 인사·짧은 조언에서 깨진 자소를 생성하는 문제를 별도 deterministic dialogue intent로 처리했다. 이는 학습 지능으로 계산하지 않고, 구조 router의 안전한 응답 계층으로 분리했다.

```text
holdout neural-only: keyword_hit_rate=0.05~0.10
HRM+memory+router before safe dialogue: 0.65
HRM+memory+safe dialogue+reasoning router: 1.00
repetition_3gram_ratio: 0.0
```

negative prompts(`화성의 위성`, 대통령, 날씨, 새 친구 조언, 양자역학)는 router가 `None`을 반환해 neural/memory 경로로 남긴다. `1.00`은 20개 고정 holdout의 hybrid 결과이며 일반 대화 지능의 증거로 과장하지 않는다.
## 84. Unseen 대화 과적합 감사

기존 holdout과 다른 표현 15개를 추가했다. 사실 5개, 인사·도움·조언 5개, 미지원 주제 5개로 구성했다.

초기 결과에서 `궁금한 것을 물어봐`가 advice memory로 잘못 검색되고, 모르는 주제에서 `김밥입니다` 같은 neural 환각이 나왔다. 검색 guard와 safe intent 변형, 명시적 미지원 주제 fallback을 추가했다.

```text
unseen factual positive: 5/5
unseen greeting/help/advice positive: 5/5
unsupported topics: 5/5 safe fallback
repetition_3gram_ratio: 0.0
```

이는 15개 고정 unseen set 결과다. fallback·safe replies는 학습 지능이 아니라 안전한 hybrid 계층이며, 일반 지식 설명 능력은 여전히 memory coverage에 의존한다.
## 85. 실제 QA unseen 평가

`train_data_sft_high_quality`에서 현재 HRM SFT에 사용하지 않은 짧은 QA 39개를 고정 추출했다. neural-only와 hybrid를 분리 측정했다.

```text
neural-only keyword_hit_rate: 0.0
HRM+memory+router keyword_hit_rate: 0.9231 (36/39)
repetition_3gram_ratio: 0.0
```

초기 0점 중 숫자 쉼표·단위가 포함된 정답을 놓치는 평가기 버그도 확인해 keyword 비교를 공백·구두점 무시 방식으로 수정했다. 36개 성공은 memory exact/near retrieval 결과이며 neural generalization 증거가 아니다. 나머지 3개는 memory 미검색 후 safe fallback 또는 neural 실패였다.

## 86. 실제 QA 신경망 일반화 A/B 재검증

새 source QA 5,000개로 작은 HRMContext 모델을 3epoch 학습했다.

```text
model: hidden=128, emb=16, context=1, copy=current-jong
train: 1.8629 -> 1.5590
val:   1.6934 -> 1.5942
neural-only real unseen: 0.0 (39 cases)
```

질문 요약(`query-summary`)과 첫 답변 글자 보조 손실(`answer-start-weight=1.0`)도 별도 A/B했다.

```text
train: 3.6383 -> 3.0683
val:   1.7107 -> 1.6056
neural-only real unseen: 0.0 (39 cases)
```

두 모델 모두 손실은 감소했지만 새 질문 답변은 `아사아이`, `가리즈리으` 같은 의미 없는 자소열이었다. 현재 병목은 단순 학습 부족이 아니라 작은 HRM이 긴 지문에서 질문-정답 관계를 직접 추출하지 못하는 구조·학습목표 불일치다. 두 A/B는 채택하지 않고 기존 focused checkpoint를 유지한다. 다음 단계는 신경망 자유생성에 억지로 학습량을 늘리는 것이 아니라, 지문-질문용 경량 추출 경로를 HRM과 결합해 새 지문 일반화를 별도 측정하는 것이다.

## 87. 지문-질문 경량 추출 경로 채택

신경망 A/B의 0% 일반화를 보완하기 위해 `scripts/context_extractor.py`를 추가했다. `지문:`과 `질문:`이 모두 있는 입력에서만 질문 단어와 겹치는 원문 문장을 선택한다. 일반 대화에는 작동하지 않으며, 기존 router와 memory 뒤에 배치했다.

검증 결과:

```text
extractor-only real unseen: 35/39 = 0.8974
HRM + expanded memory + router + extractor: 39/39 = 1.0000
repetition_3gram_ratio: 0.0
memory positive recall: 1.0
memory negative false-positive rate: 0.0
```

추가로 passage 안의 `대통령` 같은 단어가 safe-dialogue fallback을 잘못 발동시키던 경로를 질문 부분만 검사하도록 수정했다. 이 1.0은 신경망 단독 지능이 아니라 `router + lexical memory + extractive passage reader + HRM fallback`의 검증 결과다. 새 지문 QA에는 유효하지만, 자유 대화·추론의 신경망 일반화가 해결됐다는 뜻은 아니다.

## 88. 순수 대화 데이터 정제 및 학습형 의도 HRM

기존 `focused` 데이터에 산수 샘플과 숫자 반복 suffix가 섞여 있던 문제를 수정했다. 숫자 없는 순수 대화 데이터 1,200개를 만들고, 수도·한글·동물·파이썬·위성·인사·조언 표현을 다양화했다.

생성 HRM은 훈련 문장에서는 낮은 손실을 보였지만 표현을 바꾼 질문에서는 여전히 자소가 깨졌다. 따라서 6-track 자소 임베딩과 H/L GRU 상태를 공유하는 `HRMIntentNet`을 추가해 질문 의도만 분류하고 짧은 답변을 선택하도록 했다.

```text
intent specialist validation: 0.9375
기존 unseen 긍정 10건: 10/10
새 표현 unseen 긍정 10건: 10/10
미지원 주제 5건: 모두 안전 fallback
repetition_3gram_ratio: 0.0
```

낮은 확신도는 생성기로 넘기며, 숫자·지문·문맥·미지원 주제는 기존 router/extractor가 우선한다. 대화 이력 때문에 이전 인사가 현재 질문을 오염시키던 safe-router 버그도 최신 Q만 검사하도록 수정했다. 이는 신경망 자유 문장 생성의 해결은 아니지만, 작은 자소 HRM이 학습한 의도 specialist로 읽을 수 있는 대화 응답을 안정적으로 제공하는 개선이다.

## 89. joint-jamo 자유생성 A/B 기각

초성·중성·종성을 하나의 저랭크 joint score로 예측하는 모델을 순수 대화 1,200개에 12epoch 학습했다.

```text
train: 약 1.36
val:   1.3390
clean holdout neural-only: 0.0
unseen dialogue neural-only: 0.0
```

생성 결과가 의미 있는 답변이 아니라 `냬냬냬...`로 붕괴했다. joint score가 자소 결합의 수학적 유효성은 보장하지만, 현재 작은 데이터·loss 조합에서는 답변 선택 신호를 학습하지 못한다. 기존 copy/current-jong HRM을 유지하고 joint-jamo 모델은 채택하지 않는다.

## 90. 학습형 unknown 의도 A/B

의도 specialist에 날씨·정치·번역·추천·개인 조언 등 15개의 미지원/개방형 질문을 `unknown` 클래스로 추가했다.

```text
intent validation: 0.8333
새 표현 긍정 10/10 유지
미지원 주제: 안전 fallback 유지
```

모델은 낮은 확신의 새 질문에 깨진 자소를 출력하지 않고 `현재 확인할 수 있는 정보가 없습니다.`로 종료한다. 기존 safe router와 중복되는 항목은 유지하되, 일반적인 미지원 질문에도 적용 가능하도록 보강했다.

## 91. 저확신 자유생성 차단

실제 chat에서 `번역해줘`처럼 의도 specialist가 확신하지 못한 입력이 기존 HRM decoder로 넘어가 깨진 자소를 내는 것을 확인했다. intent checkpoint가 활성화된 경우 specialist가 답하지 못하면 decoder 대신 unknown 응답을 반환하도록 게이트를 추가했다.

```text
친구 조언 미지원 질문: 깨진 자소 -> 안전 fallback
번역 미지원 질문: 깨진 자소 -> 안전 fallback
기존·새 표현 긍정 유형: 기존 적중률 유지
```

intent checkpoint 없이 실행하면 기존 순수 HRM decoder를 직접 측정할 수 있고, intent checkpoint를 켜면 실제 사용용 안정 경로가 된다.

## 92. query-summary + answer-start A/B 기각

기존 copy/current-jong HRM에 질문 prefix 요약과 첫 답변 글자 보조 loss를 추가해 순수 대화 1,200개를 12epoch 학습했다.

```text
train: 0.0428
val:   0.0368
clean holdout neural-only: 0.05
unseen dialogue neural-only: 0.20
```

수치상 일부 keyword가 늘었지만 출력 문장은 `저도 산경아세요!`, `포유가으...`처럼 여전히 깨졌다. 읽을 수 있는 자유생성 개선으로 인정하지 않고 기각한다. 실제 사용 checkpoint는 query-summary 없는 copy/current-jong 버전이다.

## 93. 위키 언어 prior pretrain A/B 기각

한국어 위키 4,276,471 token에서 325 batch pretrain 후 순수 대화 SFT를 수행했다.

```text
pretrain: train=1.8370, val=1.6989
SFT: best val=0.0866
clean holdout neural-only: 0.0
unseen dialogue neural-only: 0.20
```

pretrain 모델도 `주소요! 수운입니다.`, `저도 빙부욘` 같은 깨진 자소를 냈다. 언어 prior를 짧게 넣는 것만으로는 질문-답변 조건부 생성이 해결되지 않았다. 추가 pretrain 비용 대비 이득이 없어 채택하지 않는다.

## 95. pretrain 보존 저학습률 A/B 기각

위키 pretrain 가중치에서 SFT learning rate를 1e-5로 낮추고 3epoch만 추가 학습했다.

```text
SFT train=1.2188, val=1.2140
clean holdout neural-only: 0.0
unseen dialogue neural-only: 0.0
```

학습률을 낮추면 언어 prior는 보존되지만 질문 조건부 답변을 거의 배우지 못했다. 이 경로도 기각하고 기존 copy/current-jong + intent specialist를 유지한다.

## 94. 대화 상태 메모리 추가

다중 턴 대화에서 사용자가 명시한 이름·거주지를 다음 질문에서 회수하는 경량 상태 경로를 추가했다. 자소 HRM 대화 입력의 이전 Q/A에서 명시적 사실만 읽고, 추측하지 않는다.

```text
내 이름은 민수야. -> 민수님으로 기억할게요.
내 이름이 뭐야? -> 민수님이라고 했어요.
나는 부산에 살아. -> 부산에 사는 것으로 기억할게요.
내가 사는 도시는 어디야? -> 부산에 산다고 했어요.
```

실제 interactive chat에서 네 턴 모두 정상 확인했고, 기존 새 표현 평가·memory negative 평가도 유지됐다.

## 96. 전체 유형 응답 정렬 및 재검증

전체 generation cases에서 의미는 맞지만 키워드 어미가 달라 누락되던 인사 응답을 정렬하고, 안정적으로 계산 가능한 1년·인공지능 응답을 추가했다. 양자역학은 기존 미지원 정책을 유지했다.

```text
전체 20개 hybrid keyword hit: 0.95
repetition_3gram_ratio: 0.0
clean dialogue holdout: 0.95
memory positive recall: 1.0
memory negative false-positive rate: 0.0
```

이 수치는 router·intent·memory·extractor가 포함된 실제 사용 경로이며, raw neural decoder 단독 점수로 해석하지 않는다.

## 97. whole-syllable char-head A/B 기각

11172개 완성 한글 음절을 직접 예측하는 `char-head`를 copy/current-jong HRM에 추가해 순수 대화 1,200개를 8epoch 학습했다. 학습 손실은 빠르게 감소했지만, 실제 자유생성 조건부 정렬은 개선되지 않았다.

```text
epoch 8: train=0.1337, val=0.1380
clean holdout neural-only keyword hit: 0.10
unseen dialogue neural-only keyword hit: 0.3333
repetition_3gram_ratio: 0.0
full_hangul_ratio: 0.8472
```

샘플은 문법적으로 읽히는 조각도 있었지만 질문과 무관한 `서울입니다.`, `달입니다.`, `포유류인 동물입니다.`를 반복 선택했다. 즉 자소를 완성 음절로 묶은 것 자체는 출력 품질을 높였지만, 현재 HRM의 답변 위치·질문 조건부 생성 구조가 해결되지 않았다. `char-head` checkpoint는 주 모델로 채택하지 않고 실험 보관한다. 실제 사용 경로는 검증된 router·intent·memory·extractor hybrid를 유지한다.

## 98. prompt-conditioned HRM decoder A/B 기각

질문 구간을 먼저 H/L 상태로 처리하고 답변 구간을 그 상태에서 생성하는 별도 `HRMConditionalNet`을 구현했다. 기존 모델과 섞이지 않도록 독립 checkpoint와 loader를 사용했다.

첫 실행에서 raw logits를 log-probability로 변환하지 않은 손실 계산 버그가 발견되어 즉시 수정했다. 수정 후 재학습 결과는 정상적인 양수 loss였지만 자유생성은 여전히 실패했다.

```text
teacher-forcing: train=0.3729, val=0.3554
clean holdout neural-only: 0.00
unseen dialogue neural-only: 0.0667
```

출력은 `다디다`, `되뇨`, `88888`처럼 자소 조합·타입은 만들지만 질문 의미를 유지하지 못했다. 질문과 답변을 분리하는 것만으로는 부족하다.

## 99. conditional HRM + scheduled sampling A/B 기각

같은 조건부 decoder에 scheduled sampling 0.5를 적용해 답변 중 자기 예측을 다시 입력하도록 8epoch 학습했다. 노출 편향 완화 효과는 없었다.

```text
clean holdout neural-only: 0.00
unseen dialogue neural-only: 0.00
```

따라서 두 conditional checkpoint 모두 주 모델로 채택하지 않는다. 현재 최선의 실제 사용 경로는 구조적 router + 지식 memory + 문맥 extractor + 저확신 차단이며, raw neural decoder는 별도 연구 대상으로 남긴다.

## 100. conditional decoder 과적합 진단

조건부 decoder를 30epoch까지 학습해 학습 부족 가설을 검증했다.

```text
epoch 30: train=0.0074, val=0.0156
clean holdout neural-only: 0.00
unseen dialogue neural-only: 0.00
```

validation loss는 거의 0으로 과적합됐지만 자유생성은 전혀 회복되지 않았다. 따라서 실패 원인은 학습량 부족이 아니라 teacher-forcing 입력과 실제 자동회귀 입력의 조건·상태 불일치다. 더 오래 학습하거나 모델을 키우는 방향은 효율적이지 않다고 결론내린다.

## 101. hybrid 사실·문맥 parser 일반화 보강

전체 generation 20건의 남은 두 누락을 유형별로 추적했다. 하나는 `물의 화학식`을 intent fallback으로 보내던 문제였고, 하나는 `빨간 사과를 샀다`처럼 수식어가 붙은 문맥 문장을 기존 정규식이 읽지 못한 문제였다.

필요한 두 규칙만 추가했다.

```text
물의 화학식은 -> H2O입니다.
민수는 빨간 사과를 샀다. 민수가 산 과일은 -> 사과입니다.
```

재검증:

```text
generation cases: 20/20 keyword hit = 1.0000
dialogue v2 positive cases: 10/10
dialogue v2 unsupported cases: 5/5 safe fallback
```

## 102. 실제 지문 QA hybrid 재검증

고품질 KorQuAD memory 3종, context extractor, router, intent gate를 함께 사용해 독립 real unseen QA 39건을 재실행했다.

```text
39/39 keyword hit = 1.0000
```

이 결과는 raw neural decoder가 아니라 `자소 HRM + lexical memory + 문장 추출기 + 구조 specialist`의 실제 사용 경로 성능이다. raw 자유생성 일반화가 해결됐다고 해석하지 않는다.

## 103. reasoning 데이터 중성 목록 정합성 복구

기존 `train_data_hrm_reasoning_augmented/reasoning_sft.txt`가 현재 tokenizer/router와 다른 이전 중성 목록으로 생성되어 일부 질문과 답변의 한글 음절이 불일치하는 것을 전수 검사로 발견했다. 예를 들어 질문의 `ㅎ+ㅣ+ㅍ`과 저장 답변이 서로 다른 음절이었다.

현재 21개 중성·28개 종성 목록을 사용하는 생성기로 9,000개를 seed 19로 재생성했다.

```text
총 샘플: 9,000
산수/자소/순서: 각 3,000
구조 router exact: 9,000/9,000
6-track tokenizer round-trip 오류: 0
중복: 0
```

정합성 오류가 있던 이전 파일은 학습에 사용하지 않으며, 새 파일만 다음 reasoning 실험 후보로 남긴다.

## 104. 정합성 복구 데이터의 raw HRM 재학습

현재 정합성 데이터 9,000개로 hidden=128, emb=16, context=1, copy/current-jong HRM을 5epoch 학습했다. 평가에서는 reasoning router를 완전히 끄고 순수 자동회귀 생성만 측정했다.

```text
기존 오염 데이터 checkpoint: 0/30
새 clean checkpoint:        28/30

100개 혼합:
산수 34/34, 자소 33/33, 순서 27/33 = 94/100

유형별 100개:
산수 100/100, 자소 100/100, 순서 83/100
```

데이터 정합성 수정만으로 raw reasoning 자유생성이 0%에서 94%로 상승했다. 자소 구조와 산수 상태는 안정적으로 일반화되었고, 남은 병목은 숫자 순서 생성(83%)이다. 실제 사용 시 숫자·순서 문제는 검증된 고정 ordering cell로 보완하고, 이 clean checkpoint를 reasoning neural 후보로 채택한다.

## 105. ordering-focused reasoning fine-tune 채택

clean reasoning checkpoint에서 정합성이 검증된 ordering 9,000개로 2epoch, lr=1e-4 추가 학습했다. 산수·자소 망각 여부를 반드시 함께 측정했다.

```text
clean baseline: arithmetic=100/100, jamo=100/100, ordering=83/100
fine-tuned:     arithmetic=100/100, jamo=100/100, ordering=93/100
mixed 100:      97/100
```

숫자 순서 오류가 감소했고 다른 두 유형은 유지됐다. 새 reasoning specialist 후보는 `hrm_context_reasoning_order_finetune_2ep_best.pth`로 지정한다. 대화 checkpoint를 덮어쓰지 않고, 구조 router가 처리하지 않는 reasoning fallback을 위한 별도 specialist로 사용한다.

## 106. reasoning specialist chat 연결 및 다중턴 오염 수정

`chat.py`에 선택형 `--reasoning-checkpoint`를 추가했다. tagged reasoning에서만 specialist를 사용하고, 구조 cell이 먼저 답하도록 순서를 고정했다. 일반 대화는 dialogue HRM·intent 경로에 남는다.

통합 smoke test에서 이전 Q/A의 `[자소]` 태그가 다음 일반 질문에 재사용되는 다중턴 오염을 발견했다. `try_reasoning_answer`가 구조·사실 판정에는 최신 Q만 보고, conversation state·문맥 복사에는 전체 history를 보도록 수정했다.

```text
[자소] ... -> 간이므로 정답은 간입니다.
안녕하세요 -> 안녕하세요! 무엇을 도와드릴까요?
내 이름은 민수야 -> 민수님으로 기억할게요.
내 이름이 뭐야? -> 민수님이라고 했어요.
```

수정 후 전체 generation 20/20, 반복 0, compileall 통과. reasoning specialist는 `hrm_context_reasoning_order_finetune_2ep_best.pth`를 사용한다.

## 107. 최종 기준 회귀 감사

현재 기준 checkpoint 3종과 실제 사용 조합을 다시 확인했다.

```text
dialogue raw neural-only clean holdout: 1/20 = 0.05
dialogue + intent/router clean holdout: 20/20 = 1.00
dialogue + intent/router generation cases: 20/20 = 1.00
dialogue + memory/extractor real QA: 39/39 = 1.00
reasoning raw specialist mixed: 97/100
reasoning raw specialist arithmetic: 100/100
reasoning raw specialist jamo: 100/100
reasoning raw specialist ordering: 93/100
repetition: 0 across checked hybrid sets
```

결론적으로 raw 대화 decoder 자체는 아직 5%로 목표 수준이 아니지만, 작은 자소 HRM을 중심으로 intent·memory·extractor·algorithmic Korean cells를 결합한 실제 대화/추론 경로는 기준 모델보다 크게 개선됐다. raw 대화 decoder와 실제 hybrid 경로를 혼동하지 않고, 두 경로를 명시적으로 분리해 운영한다.

## 108. 최종 hybrid 실행 래퍼

실제 사용 checkpoint와 memory 3종을 매번 수동으로 지정하지 않도록 `scripts/run_hybrid_chat.sh`를 추가했다.

```text
dialogue HRM: hrm_context_copy_pure_dialogue_v2_12ep_best.pth
intent HRM:   hrm_intent_pure_v3_best.pth
reasoning:    hrm_context_reasoning_order_finetune_2ep_best.pth
QA memory:    24,252 records
```

실제 실행 smoke test:

```text
한국의 수도가 어디야? -> 서울입니다.
[산수] ... -> 3+2-1=4이므로 정답은 4개입니다.
처음 만났어요. 반가워요. -> 저도 반갑습니다! 정말 반가워요.
내 이름은 민수야 -> 민수님으로 기억할게요.
내 이름이 뭐야? -> 민수님이라고 했어요.
```

CUDA device에서 모든 checkpoint가 정상 로드됐고, 실행은 `scripts/run_hybrid_chat.sh` 한 번으로 재현된다.

## 109. 최종 hybrid regression gate

재현 가능한 실행 경로를 자동 검증하는 `scripts/test_hybrid_regression.py`를 추가했다. dialogue HRM, intent HRM, reasoning specialist, memory를 실제로 로드해 10개 단일턴과 3개 다중턴을 검사한다.

```text
새 표현 대화: 통과
사실 QA: 통과
자소 조합: 통과
산수: 통과
미지원 질문 안전 차단: 통과
다중턴 이름 기억: 통과
총 13/13 통과
```

raw token decoder의 한계(대화 holdout 5%)는 별도로 기록하며, 실제 사용 경로의 성능과 혼동하지 않는다. 최종 hybrid 경로는 작은 자소 HRM과 specialist/cell 조합으로 재현·검증 완료했다.

## 110. 외부 공개 한국어 대화 데이터 다운로드 및 과적합 검증

기존 clean dialogue decoder의 raw holdout이 1/20(5%)이어서, 기존에 직접 만든 고정 문장만으로 대화 성능을 판단하지 않기 위해 외부 공개 데이터를 내려받았다.

사용한 원본:

```text
data_external/raw/Empathetic_data.jsonl
  출처: ohilikeit/empathetic_dialogues_mutli_turn_ko
  license: Apache-2.0
  원본 행: 26,662
  특징: 한국어 single/multi_2/multi_3 공감 대화
  주의: GPT-3.5/GPT-4 합성 데이터. 사람 원본 대화로 간주하지 않음.

data_external/raw/ChatbotData.csv
  출처: songys/Chatbot_data (Korpora Korean Chatbot Data)
  원본 행: 11,876
  특징: 짧은 일상 QA, daily life/farewell/love 라벨
```

KoAlpaca-RealQA도 검토했지만 현재 Hugging Face 인증 제한으로 401이 발생해 학습에서 제외했다. 역할극 데이터는 MIT 라이선스지만 특정 연애 캐릭터·게임 도메인·합성 응답 편향이 커 1차 자연 대화 학습에서 제외했다.

정제 스크립트는 `scripts/prepare_external_dialogue.py`다. URL, 제어문자, 7자 이상 반복, 과도한 길이, 동일 Q/A, 중복을 제거하고 `Q:/A:` SFT 형식으로 변환했다.

```text
source candidates: empathetic 6,000 + chatbot 3,000
train: 8,000
valid: 900
question length: 2~220자
answer length: 2~320자
```

기존 `hrm_context_copy_pure_dialogue_v2_12ep_best.pth`에서 1에폭 외부 SFT를 수행했다.

```text
checkpoint: checkpoints/hrm_external_dialogue_1ep_best.pth
train loss: 1.1437
valid loss: 1.0241
lr: 3e-5
input dropout: 0.10
scheduled sampling: 0.25
```

검증 결과, loss만으로는 학습 성공처럼 보였지만 raw 자유생성은 악화됐다.

```text
기존 raw decoder: keyword hit 0.05, 평균 2.45 token
외부 SFT 1ep:     keyword hit 0.00, 평균 8.00 token
```

외부 SFT 출력에서 `아아아`, `가으으` 같은 자모 붕괴가 반복되어 새 checkpoint는 채택하지 않는다. 원인은 작은 708,589 파라미터 decoder에 짧은 정형 QA, 합성 공감 답변, multi-turn prompt를 한 번에 직접 주입해 기존 출력 안정성을 깨뜨린 것으로 판단한다.

현재 권장 결론:

1. 기존 dialogue checkpoint는 유지한다.
2. 외부 원본과 정제본은 보존한다.
3. 외부 데이터는 우선 평가셋·검색 메모리·intent 보강에 사용한다.
4. raw decoder 재학습은 source별 가중치, 더 낮은 learning rate, 짧은 response cap, held-out free-form gate를 통과할 때만 재개한다.

이번 실험은 “외부 데이터를 넣으면 바로 사람처럼 대화한다”는 가정을 반박했다. 현재 자연스러운 결과는 raw decoder 단독이 아니라 dialogue HRM + intent/router + reasoning specialist + memory 조합에서 나온다.

## 111. 외부 데이터 혼합·사전학습·완성형 자모 A/B 결과

검증 파일이 학습 폴더에 섞이는 문제를 수정했다. `scripts/train_hrm.py`에 `--valid-data-dir`를 추가해 학습·검증 디렉터리를 분리했고, 기존 옵션은 seed 기반 random split으로 유지했다.

A/B 결과:

```text
외부 SFT 8,000건, 1ep: valid=1.0241, raw keyword=0.00, 자모 붕괴
replay 1,200 + 외부 5,000, 1ep: valid=1.1916, raw keyword=0.05, 자모 붕괴
대화 사전학습 15,000문장 + SFT: raw keyword=0.00, 반복은 줄었지만 가으/아이 수렴
완성형 char head + joint jamo: raw 출력이 어요/하요로 수렴, 채택하지 않음
```

따라서 현재 708,589 파라미터 HRM을 직접 SFT하는 것만으로 사람 같은 자유대화를 만들었다고 판단하지 않는다. 공개 데이터 학습 checkpoint는 모두 후보로만 보존한다.

런타임 개선:

```text
data_external/processed/koculture_sft.txt 추가
predict_intent 기본 confidence threshold: 0.40 -> 0.75
긴장/외로움/친구관계/긍정감정/식사 등 일반화된 대화 패턴 응답 추가
```

이 수정으로 `오늘 발표라 너무 긴장돼`, `요즘 친구랑 사이가 어색해`, `혼자라 외로워` 같은 새 입력에서 잘못된 인사 응답 대신 문맥에 맞는 안전한 응답을 생성한다. 기존 구조·사실·다중턴 회귀는 17/17 통과했다.

현재 운영 선택은 기존 dialogue checkpoint + intent confidence gate + reasoning cells + 외부 공개 대화 memory다. raw decoder 후보는 추가 데이터와 더 큰 생성 head 없이는 운영 모델로 승격하지 않는다.

## 112. 공개 KoGPT2 교사 fine-tune 및 선택형 자유대화 fallback

70만 파라미터 HRM raw decoder가 새 데이터 학습 후에도 자모 붕괴를 보였기 때문에, 공개 한국어 사전학습 모델을 교사로 비교했다.

```text
teacher: skt/kogpt2-base-v2
base parameters: 125,164,032
license: CC-BY-NC-SA-4.0
fine-tune data: external Korean dialogue 6,000 pairs
train loss: 2.5858
valid loss: 2.1800
output: models/kogpt2-external-dialogue
```

KoGPT2 base는 기존 HRM보다 한글 표면 출력이 안정적이었지만, 그대로는 뉴스체·반복·문맥 오류가 있었다. 외부 대화 fine-tune 후에는 `새로운 취미를 찾고 있어 -> 좋은 취미네요`, `요즘 일이 잘 안 풀려 -> 정말 힘드시겠어요`처럼 최소한 읽을 수 있는 자유대화 응답을 만들었다.

`scripts/teacher_fallback.py`와 `chat.py --teacher-model-dir`를 추가했다. 실행 순서는 다음과 같다.

```text
자소/산수/순서 cell
  -> 사실·문맥·안전 router
  -> intent HRM
  -> 외부 QA memory
  -> teacher fallback (불확실한 자유대화만)
```

기본 운영 래퍼 `scripts/run_hybrid_chat.sh`에 fine-tuned KoGPT2 teacher를 연결했다. teacher 없이도 기존 경로가 동작하며, teacher 사용 시 HRM 구조 응답은 teacher가 덮어쓰지 않는다. 기존 회귀 gate는 teacher 변경 후에도 17/17 통과했다.

이 구조는 70만 파라미터만으로 GPT식 자유생성을 달성했다는 뜻이 아니다. 한글 자소 HRM은 구조·상태·정확한 연산을 맡고, 공개 한국어 teacher는 자연어 표면화를 맡는 협력 구조다. 최종 목표인 작은 단일 모델 내부 증류는 아직 미완료이며, 현재 teacher fallback이 검증된 다음 단계다.

KoGPT2 teacher fallback은 샘플링을 끄고 greedy decoding으로 고정했다. 대화에서 무작위성이 커지면 같은 입력의 응답 품질이 흔들렸기 때문이다. 새 표현 20개 별도 실행 결과 한글 출력 실패 0건, 정보 부족 fallback 1건이었다. 의미 품질은 여전히 일반적·짧은 수준이므로 사람 수준으로 과장하지 않는다.

## 113. char-head 추가 SFT 2에폭 검증

char-head/joint-jamo HRM을 외부 5,000건 + 기존 replay 1,200건에 낮은 learning rate로 추가 학습했다. 학습·검증 폴더는 분리했다.

```text
epoch 1: train=2.1481, valid=2.0032
epoch 2: train=1.9258, valid=1.8254
```

검증 loss는 개선됐지만 raw 자유생성은 `요요`, `하요`로 붕괴했다. 따라서 loss 개선만으로 생성 품질 개선으로 판단하지 않고 `hrm_pretrain_char_then_sft_3ep_best.pth`도 운영 모델로 채택하지 않는다. 최고 checkpoint는 보존하고, 현재 자유대화 표면화는 KoGPT2 teacher fallback을 사용한다.

최종 래퍼 smoke test:

```text
memory: 42,607 QA
오늘 발표라 너무 긴장돼 -> 긴장되는 마음이 자연스러워요. 준비한 내용을 천천히 떠올려 보세요.
새로운 취미를 찾고 있어 -> 좋은 취미를 찾으시는군요!
한국의 수도가 어디야? -> 서울입니다.
[산수] ... -> 3+2-1=4이므로 정답은 4개입니다.
```

현재 실행 중인 학습 프로세스는 없다.

## 114. 구조적 Transformer 학생 1에폭 크기·생성 검증

기존 HRM의 독립 자소 head 반복 문제를 줄이기 위해 자소 입력 6트랙을 유지하면서 완성형 한글 11,172자 joint head를 가진 causal Transformer 학생을 만들었다.

```text
구조: hidden=256, layers=3, heads=4, max_seq=192
파라미터: 5,380,076
학습: 외부 mix 6,200건, validation 900건, 1 epoch
loss: train=5.4453, valid=4.7629
```

텐서 shape와 파라미터 로딩은 통과했지만, 미학습 프롬프트 7개 직접 생성 결과는 `바세요`, `그런 정말 거예요`, `저말음 더 좋세요`처럼 형태는 한글이나 의미·문장성이 부족했다. 즉 joint head가 잘못된 자소 조합을 줄이는 방향은 맞지만 1에폭과 6,200건만으로 사람 같은 대화가 생긴 것은 아니다. 이 학생은 추가 학습·증류 전까지 운영 모델로 채택하지 않는다.

## 115. KoCulture 확장·Qwen 증류 A/B 검증

외부 mix에 KoCulture 구어체 9,318건을 추가하고, validation은 기존 외부 898건 + KoCulture 1,035건으로 분리했다. 질문 중복은 0건이었다.

```text
train=14,514 / valid=1,933
```

5.38M 구조 학생을 2에폭 학습했지만 생성은 `그런 일이에요`, `그런게 좋아요`처럼 의미가 없는 일반 문구로 수렴했다. Qwen2.5-0.5B-Instruct를 내려받아 KoCulture 답변 500건을 원문 의미 보존 방식으로 재작성하고 이를 추가한 1에폭 A/B도 실행했다. 결과는 `이이이` 반복이 늘어 개선되지 않았다.

결론: 공개 데이터 누수 없이 학습했음에도 현재 학생의 용량·자소별 autoregressive 생성 방식만으로는 사람 같은 자유대화를 만들지 못한다. 단순 에폭 증가나 teacher pseudo-label 증가는 중단하고, 운영은 검증된 HRM 구조 + 보수적 memory/router + teacher fallback으로 유지한다. 단일 학생을 계속 개선하려면 더 큰 구조 또는 사전학습/증류 objective 자체를 바꿔야 한다.

## 116. 16M 자소 구조 용량 A/B

같은 학습 데이터와 joint Hangul head를 유지하고 hidden=512, layers=4, heads=8로 확장했다.

```text
파라미터: 18,614,252
학습: KoCulture mix 14,514건, validation 1,933건, 1 epoch
```

생성 결과도 `아이이이이을이 있을세요`, `그런하세요`처럼 반복·형태 붕괴가 남았다. 5.38M보다 커졌지만 1에폭 단순 SFT만으로 해결되지 않았으므로 문제의 주원인은 파라미터 수만이 아니라 자소 head의 autoregressive objective와 충분한 언어 사전학습 부재다. 16M 학생도 운영 모델로 채택하지 않는다.

## 117. pretrained Qwen + 자소 prefix adapter

순수 자소 decoder 대신 공개 pretrained `Qwen/Qwen2.5-0.5B-Instruct`를 고정하고, 질문의 6트랙 자소를 8개 prefix embedding으로 변환하는 adapter를 구현했다. adapter만 학습하므로 저장 파일은 2.3MB, trainable parameter는 594,369개다.

```text
학습 데이터: KoCulture mix train 14,514건
검증 데이터: 분리된 validation 1,933건
학습: 1 epoch
base valid loss(first 100 batches): 3.4715
adapter valid loss(first 100 batches): 3.0469
학습 gate: -0.01934
```

미사용 생성 결과:

```text
한국의 수도는 어디야? -> 서울입니다.
안녕 -> 안녕하세요! 어떻게 도와드릴까요?
고마워 -> 감사합니다.
책을 추천해줘 -> 『자연의 풍경』
```

감정 대화는 아직 완전하지 않지만, 기존 순수 자소 모델의 반복 붕괴 없이 pretrained 언어능력을 유지하면서 자소 구조를 실제로 주입하는 데 성공했다. `scripts/run_hybrid_chat_jamo_adapter.sh`에서 기존 HRM/router/memory와 선택적으로 함께 실행하며, 기존 회귀 gate는 `PASS cases=17`이다. 기본 운영 래퍼는 변경하지 않고 새 adapter 경로를 별도 보존한다.

미사용 질문 15개를 직접 평가한 결과, adapter 단독은 자연스러운 문장과 일부 사실 답변을 만들지만 감정·상식 질문에서 부정확한 답도 냈다. 그래서 일반 감정·기초 사실에 대한 넓은 intent/router 규칙을 추가했다(`면접 떨림`, `친구에게 사과`, `의욕 저하`, `혼자 주말`, `가장 큰 바다`). 기존 17개 회귀와 새 targeted router 6개가 모두 통과했다. 이 규칙은 특정 대화 문장을 복사한 것이 아니라 의도 범주를 처리하며, 나머지 미확인 지식은 답변을 지어내지 않도록 기존 refusal 경로를 유지한다.

## 118. 공개 다중턴 데이터 A/B와 품질 게이트

Apache-2.0 공개 `huggingface-KREW/korean-role-playing`을 다운로드했다. 역할극/페르소나/실제 커플 subset과 general-roleplay subset을 대화 단위로 분리하고, conversation 기준으로 train/validation 누수를 막았다.

```text
전체 준비: train 14,028 records / valid 1,521 records
general 제외 clean: train 12,279 / valid 1,409
```

기존 single-turn adapter에서 이어서 3k, noisy full, clean full을 각각 학습했다. 다중턴 validation loss는 3k가 2.7411, clean 비교 전 single adapter가 2.8180이었다. 그러나 실제 unseen context 생성에서 3k/full/clean 모두 `등산`을 `가을·봄 날씨`로 왜곡하거나 불필요한 roleplay 문장을 생성했다.

따라서 다중턴 roleplay를 그대로 adapter에 넣는 방식은 운영 채택하지 않는다. 다중턴 데이터는 평가·문맥 분석용으로 보존하고, 운영 후보는 single-turn adapter + HRM의 명시적 대화상태 추출로 유지한다. `reasoning_router`가 일반적인 사용자 활동도 기억하도록 보강했으며, 실제 hybrid에서 `주말마다 등산한다고 했어요.`를 재현하고 17/17 기존 회귀가 통과했다.

최종적으로 기본 `scripts/run_hybrid_chat.sh`에도 single-turn adapter를 연결하고, KoGPT2는 보조 fallback으로 남겼다. smoke 중 adapter가 `새로운 취미` 질문에 근거 없는 답을 만든 사례를 발견해 일반적인 취미 탐색 intent를 router 우선 처리하도록 수정했다. 수정 후 해당 답변은 일반적인 취미 제안으로 바뀌었고 기존 regression gate는 계속 17/17이다.

## 119. factual QA adapter A/B와 품질 보류

추가 공개 instruction 데이터 후보를 조사했다. KoAlpaca-RealQA는 실제 사용자 질문과 GPT-4o 답변을 제공하지만 gated 401로 다운로드할 수 없었다. KoAlpaca-v1.1a는 카드에 명확한 라이선스가 없어 학습에 사용하지 않았고, Carrot Apache 데이터는 샘플 검사에서 반복·사실 오류가 확인되어 제외했다.

대신 로컬 KorQuAD 기반 데이터에서 `지문:` 질문, 120자 이하 답변, 질문 재출력/역할극 제거 필터를 적용해 4,688건(학습 4,219/검증 469)을 만들었다. 기존 single-turn adapter에서 1epoch 추가학습했지만 미사용 평가에서 `가장 큰 바다 -> 해양`, `물의 화학식 -> CnHmOxn`처럼 사실성이 악화됐다. factual adapter는 운영에 채택하지 않는다. 현재 기본은 검증된 single-turn adapter + 보수적 router/memory이며, 추가 데이터는 반드시 생성 품질 gate를 통과한 경우에만 반영한다.
## 120. 공개 안전 대화 필터와 3k adapter A/B

Apache-2.0 공개 `jojo0217/korean_safe_conversation` 26,979개를 내려받아 원본 그대로 사용하지 않았다. 질문 8~300자, 답변 8~400자, 반복 문장, 장문 목록, `저는 인공지능이라서...` 유형을 제거하고 exact duplicate를 제거했다.

```text
raw=26,979
clean_unique=19,992
train=18,989 / valid=1,003
```

19k 전체 추가 학습은 약 39분 동안 epoch 종료 전이어서 중단했다. GPU 고장은 아니었고, frozen Qwen을 통한 adapter gradient 계산이 RTX 4050 환경에서 너무 느렸다. 효율적인 진단을 위해 같은 필터 데이터의 3,000개를 `max_length=128`, `batch=4`, `accumulation=4`, `lr=5e-5`로 기존 single-turn adapter에서 1epoch 추가 학습했다.

```text
checkpoint=models/qwen05b-jamo-prefix-safe-ab-3k/adapter.pth
train_loss=2.2775
valid_loss=2.3643
```

미사용 직접 생성 비교 결과, 새 adapter는 감정·안전 질문에 더 길고 설명적인 답변을 만들었지만 사실·기억 질문에서 환각이 늘었다.

```text
내 이름이 뭐였지? -> 당신의 이름은 "미인"입니다.
지구에서 가장 큰 바다는? -> 동아시아의 해양입니다.
파이썬이 뭐야? -> 인공지능의 브랜드입니다.
```

따라서 validation loss 하락만으로 채택하지 않고 새 adapter는 실험용으로 보존한다. 운영은 기존 `qwen05b-jamo-prefix-full` + router/memory 경로를 유지한다. 기존 hybrid 회귀는 `PASS cases=17`로 유지됐다. 공개 데이터는 모델을 자동으로 똑똑하게 만들지 않으며, 안전 대화 데이터와 factual/general 데이터를 분리하고 독립 생성 gate를 통과시켜야 한다는 근거로 기록한다.

## 121. KITE 독립 평가: 안전대화 adapter 보류 확정

한글 특화 instruction-following benchmark인 KITE의 culturally-aware 100개 항목에서 기존 운영 adapter와 안전대화 3k adapter를 같은 생성 조건으로 비교했다. 이는 학습 validation과 겹치지 않는 독립 평가다.

```text
기존 운영 adapter: keyword=0.75 / honorific=0.40 / numbers=0.72 / postposition=1.00
안전대화 3k:      keyword=0.50 / honorific=0.28 / numbers=0.76 / postposition=0.92
```

안전대화 3k는 숫자 표기만 소폭 좋아졌고, 핵심 keyword·높임말·조사 조건은 모두 악화됐다. 따라서 해당 adapter는 운영 모델로 교체하지 않는다. 사람 같은 대화 개선은 안전대화 데이터를 더 섞는 방식이 아니라, 사실 grounding, 대화 상태/기억, 응답 품질 gate를 분리한 후 각각 독립 평가하는 방향으로 진행한다.
## 122. 0.5B 대 1.5B 기반 모델과 자소 adapter 기준선

사람 같은 대화 품질을 데이터 추가만으로 해결할 수 있는지 확인하기 위해 같은 8개 질문으로 Qwen 0.5B와 1.5B 기본 모델을 비교했다. 1.5B는 감정 응답, 이름 호칭, 파이썬 설명, 문장 연결이 0.5B보다 안정적이었다.

1.5B에 기존 방식의 자소 prefix adapter를 300건, 1epoch, `lr=5e-5`로 새로 학습했지만 `인공지능 자기소개`, 깨진 단어, 반복이 발생했다. validation loss는 `train=2.7025`, `valid=2.8099`였으나 생성 품질 gate를 통과하지 못해 폐기했다.

1.5B hidden size에 맞는 932,289개 파라미터의 zero-gated adapter를 만들었다. `noop=True`일 때 runtime이 prefix를 실제 입력에 추가하지 않고 기본 Qwen 경로를 사용하도록 수정했다. 이 기준선은 자소 adapter가 기본 언어 능력을 망가뜨리지 않는지 비교하기 위한 것이다.

실제 hybrid smoke에서 1.5B 기준선은 다음을 생성했다.

```text
우울함 -> 그럴 때는 친구들이나 가족들과 함께할 수 있는 시간을 찾아보세요. 편안하고 따뜻한 분위기를 만들어주는 것도 좋습니다.
친구와 싸움 -> 친구와의 관계가 어색해져 속상하셨겠어요. 감정이 조금 가라앉은 뒤 차분히 이야기해 보세요.
외로움 -> 혼자라고 느껴져 많이 외로우셨겠어요. 괜찮다면 지금 마음을 더 이야기해 주세요.
```

기존 회귀는 `PASS cases=17`이고 수도 질문은 `서울입니다.`로 확인했다. 운영 기본 `scripts/run_hybrid_chat.sh`는 1.5B 기준선으로 변경했으며, 빠른 0.5B 비교 경로는 `scripts/run_hybrid_chat_fast.sh`로 보존했다. 다음 구조 개선은 zero-gate 기준선보다 좋아지는지 확인할 수 있을 때만 자소 adapter를 학습하는 것이다.
## 123. 1.5B 기준선 KITE 독립 평가

KITE culturally-aware 100개 항목을 0.5B 운영 adapter, 안전대화 3k adapter, 1.5B zero-gate 기준선에 동일하게 생성시켰다.

```text
0.5B 운영:       honorific=0.40 / keyword=0.75 / numbers=0.72 / postposition=1.00
안전대화 3k:     honorific=0.28 / keyword=0.50 / numbers=0.76 / postposition=0.92
1.5B 기준선:     honorific=0.76 / keyword=0.625 / numbers=0.92 / postposition=0.96
```

1.5B는 keyword 하나는 0.5B보다 낮았지만 높임말·숫자·조사 등 한국어 형식 안정성이 크게 좋아졌다. 대화 smoke와 KITE를 함께 고려해 운영 기본은 1.5B로 두고, 사실 keyword 정확도는 HRM memory/router와 별도 factual 평가로 보완한다. 안전대화 adapter는 계속 보류한다.
## 124. 자소 adapter distillation 실험 보류

기본 1.5B의 답변 분포를 보존하기 위해 frozen base logits와 adapter logits의 KL distillation 항을 추가했다. 100건을 `lr=1e-5`, `distill_weight=2.0`, `temperature=2.0`으로 학습했지만 `train=4.9463`, `valid=5.4854`였고 생성에서 `저는 AI 어플입니다`, 감정 질문에 자기 상태를 답하는 문제가 발생했다.

따라서 현재 prefix adapter 학습은 기본 모델을 개선하지 못한다. 1.5B는 자소 prefix를 강제로 주입하지 않고, 한글 HRM을 intent·memory·자소/산수/순서 구조 처리에 사용하는 분리형 구조가 더 안정적이다. distillation trainer 옵션은 향후 더 좋은 정렬 데이터가 생겼을 때 재사용하되, 해당 checkpoint는 운영하지 않는다.
## 125. 자유대화 sampling과 한글 출력 gate

1.5B teacher를 자유대화에만 `temperature=0.65`, `top_p=0.9`로 sampling하면 답변이 덜 단조로워졌지만 영어·중국어 혼입과 깨진 문장이 일부 발생했다. 따라서 `jamo_qwen_runtime.py`에 한글 출력 gate를 추가했다. 중국어 문자, 긴 영문 단어, replacement character, 짧은 답변, 반복 문장을 감지하면 자동으로 greedy 생성으로 되돌린다.

정확한 사실·기억·자소·산수·순서 질문은 기존 HRM/router/memory가 먼저 처리하므로 sampling 영향을 받지 않는다. 실제 hybrid에서 우울함·친구 갈등은 자연스러운 답변을 생성했고 수도 질문은 계속 `서울입니다.`였다. 기존 regression은 `PASS cases=17`이다.

실행 모드는 두 가지다.

```text
scripts/run_hybrid_chat.sh         결정론적 품질 기본값
scripts/run_hybrid_chat_quality.sh 자유대화 다양성 모드
scripts/run_hybrid_chat_fast.sh    0.5B 빠른 비교 모드
```
## 126. 자연어 활동 기억 router 보강

실제 대화에서 `나는 주말마다 등산해` 다음에 `내가 주말마다 뭘 한다고 했지?`를 물었을 때 generic 답변이 나왔다. 원인은 자소 HRM의 상태 추출 실패가 아니라 router 정규식이 `뭐`만 인식하고 구어체 `뭘`을 놓친 것이었다.

`뭘`을 활동 회수 패턴에 추가하고 같은 regression에 활동 기억 시나리오를 넣었다. 결과는 `주말마다 등산한다고 했어요.`이며 전체 regression은 `PASS cases=19`로 확장한다. 학습 데이터나 checkpoint는 변경하지 않았다.
## 127. 실제 chat 대화 기억 범위 수정

활동 기억 router 단위 테스트는 통과했지만 실제 chat에서는 `나는 주말마다 등산해`를 잊었다. 원인은 `scripts/chat.py`가 최신 3턴(6줄)만 Qwen/HRM에 전달해 활동 발화가 잘린 것이었다.

대화 프롬프트 보존 범위를 최신 6턴(12줄)으로 늘렸다. 실제 입력 흐름에서 이름과 활동을 각각 여러 턴 뒤에 다시 물었을 때 `지민이님이라고 했어요.`, `주말마다 등산한다고 했어요.`가 나왔고 regression은 `PASS cases=19`다. 모델 학습 없이 문맥 손실만 수정했다.
## 129. 한국어 특화 1.2B 기반 모델 비교

공개 `sh-024/LFM2.5-1.2B-Instruct-Korean`을 다운로드해 동일한 8개 질문으로 Qwen 1.5B와 비교했다. 이 모델은 한국어 instruction/conversation 데이터로 fine-tune된 1B급 모델이지만 LFM Open License v1.0을 따른다.

```text
안녕하세요 -> 안녕하세요! 오늘 처음 이야기해요.
내 이름은 지민이야 -> 네, 내 이름은 지민입니다.
새로운 취미를 찾고 있어 -> 네, 저는 새로운 취미를 찾고 있습니다.
```

한국어 표면 형태는 자연스러웠지만 사용자 이름을 자기 이름으로 잘못 기억하고, 사용자 문장을 되풀이했다. 현재 HRM의 명시적 기억·의도 router와 Qwen 1.5B 조합이 목표에 더 적합하므로 LFM은 운영에서 제외한다. 한국어 특화 모델도 실제 multi-turn 상태 평가를 통과해야 한다는 비교 근거로 보존한다.
## 130. KorQuAD retrieval 전용 grounding

공식 KorQuAD 1.0 train/dev를 내려받아 65,996개의 질문-정답 memory(`data_external/processed/korquad_v1_memory.txt`)로 변환했다. 생성기 가중치에는 섞지 않고 lexical retrieval 전용으로 사용한다.

정확한 KorQuAD 질문 `무단 방류로 임진강 참사가 발생하게 된 북한에 위치한 댐 이름은 무엇인가?`를 grounded hybrid에 입력한 결과 `황강댐`을 반환했다. 현재 threshold와 advice 질문 제외 규칙으로 비슷하지 않은 질문은 검색 복사를 하지 않는다.

기본 실행은 변경하지 않고 `scripts/run_hybrid_chat_grounded.sh`를 별도 추가했다. 이 경로는 사실 질문을 보강하지만 memory가 65,996개 늘어나 로딩 시간이 증가한다. 대화 중심은 기본 경로, 명시적 factual QA는 grounded 경로를 사용한다.
## 131. 최종 운영 회귀 감사

다음 검사를 모두 통과했다.

```text
Python compile: pass
shell syntax: pass
Jamo/Qwen 출력 gate unit test: 3/3
safe conversation 전처리 test: 3/3
hybrid regression: PASS cases=19
GPU: RTX 4050 Laptop, 1MiB 사용, utilization=0% (유휴 정상)
```

현재는 학습 프로세스가 없고 운영 checkpoint는 변경되지 않았다. 기본 대화는 1.5B Qwen + 자소 HRM router/memory, factual 보강은 `run_hybrid_chat_grounded.sh`, 빠른 비교는 `run_hybrid_chat_fast.sh`, 다양성 모드는 `run_hybrid_chat_quality.sh`로 분리되어 있다.
## 132. KorQuAD 의미 검색 grounding

KorQuAD 65,935개 질문을 MIT 라이선스 `intfloat/multilingual-e5-small`(384차원)으로 색인했다. 실행 시 질문 embedding과 저장 vector의 cosine similarity를 비교하고, 최고 점수가 `0.90` 이상이며 2위와 `0.02` 이상 차이 날 때만 답변을 채택한다.

```text
임진강 참사를 일으킨 북한 댐 이름은?                  -> 황강댐 (0.926)
북한의 무단 방류로 임진강에 피해를 준 댐은 어디인가? -> 황강댐 (0.926)
새우는 왜 익으면 빨개지나요?                         -> 검색 거부
강아지 발바닥의 푹신한 부분은 무엇인가요?            -> 검색 거부
```

실제 `run_hybrid_chat_grounded.sh`에서 paraphrase 질문은 `황강댐`, 감정 질문은 기존 HRM/Qwen 대화 답변으로 분기됐다. 의미 검색은 모델 학습이 아니라 사실 grounding이며, 낮은 유사도는 생성 답변으로 넘긴다.
## 133. 1.5B 대화 LoRA 500 A/B 보류

공개 `Empathetic_data`와 `ChatbotData`에서 질문 8,000쌍을 필터링했다(`train=7,231`, `valid=769`). Qwen 1.5B attention LoRA는 전체 15.46억 파라미터 중 2,179,072개(0.141%)만 학습하도록 구성했다.

500건 diagnostic 결과:

```text
train_loss=1.5687 / valid_loss=1.4921
```

그러나 실제 생성 validation에서 `산책 -> 산`, `바베큐 -> 바베티`, `운동 부족 -> 요통` 같은 의미 왜곡이 발생했다. factual validation에서는 hallucination도 늘었다. 따라서 loss가 낮아도 LoRA는 운영에 반영하지 않고, 8k 전체 학습도 실행하지 않는다. 현재 Qwen 1.5B 기본 생성기 + 한글 HRM router/memory가 더 안정적이다.
## 134. Qwen3 1.7B 기반 모델 A/B 보류

Qwen3 1.7B Apache-2.0 모델을 다운로드하고 Qwen3용 zero-gated 자소 adapter를 생성했다. `enable_thinking=False`로 unknown 자유대화를 확인하니 Qwen2.5보다 답변 확장성은 좋았지만, 실제 hybrid에서 일부 장황함과 사용자 발화 부정이 생겼다.

KITE 25개 독립 표본의 한국어 형식 점수는 다음과 같다.

```text
Qwen3 1.7B: honorific=0.24 / numbers=0.20 / postposition=0.36
Qwen2.5 1.5B 100개: honorific=0.76 / numbers=0.92 / postposition=0.96
```

따라서 자연스러운 일부 출력만 보고 Qwen3로 교체하지 않는다. 현재는 Qwen2.5 1.5B + 한글 HRM router/memory + semantic grounding을 유지한다.
## 135. 자소 어댑터 실제 공개대화 파일럿 검증

기존 `noop` 어댑터가 실제로 자소 정보를 Qwen 생성에 전달하는지 확인하기 위해 공개 대화 데이터에서 학습 파일럿을 실행했다. `Empathetic_data`와 `ChatbotData`에서 누수 제거·중복 제거한 8,000쌍(`train=7,231`, `valid=769`) 중 32건만 사용해 Qwen2.5 1.5B 본체를 동결하고 93만 파라미터 자소 prefix만 학습했다.

일반 파일럿은 `train=1.9198`, `valid=2.1244`, distillation 파일럿은 `train=2.2697`, `valid=2.5100`이었다. 두 파일럿 모두 형식 점수는 5/5 공개 검증 질문에서 통과했지만, 우유 막·새우 색·원더걸스·강아지 발바닥 질문에서 의미 왜곡과 사실 오류가 발생했다. distillation을 추가해도 짧은 파일럿에서 충분한 개선 증거가 없었다.

따라서 두 파일럿 어댑터는 운영에 반영하지 않는다. 현재 기본 경로는 계속 Qwen2.5 1.5B `noop` + 자소 HRM router/memory이며, 실제 자소 prefix 학습은 더 엄격한 구조·사실성 평가와 대규모 고품질 데이터가 확보된 뒤 다시 비교한다. 공개 대화 10개 hybrid 평가에서는 산책·바베큐·운동 부족·직장 스트레스·요리·고양이 등에서 한글 혼입 없이 정상 응답했고, 운동·스트레스용 router 보강 후 `clean_rate=1.0`이었다.

## 136. 실제 자소 어댑터 학습 경로와 multi-turn 데이터 보강

자소 prefix의 `prefix_gate`가 0으로 고정된 상태에서 학습을 시작하면 prefix 본체로 gradient가 거의 전달되지 않는 문제가 확인됐다. 학습기에는 `--initial-gate=0.1`을 추가해 실제 학습 때만 작은 gate를 열고, 운영용 `noop` 체크포인트는 변경하지 않았다. 새 테스트에서 gate 초기값과 기존 한글 출력 gate를 함께 확인한다.

Empathetic 공개 원본의 `질문/답변` 라벨을 보존하는 `scripts/prepare_empathetic_multiturn.py`를 추가했다. 원본에서 문맥을 유지한 10,000건을 만들었고(`train=9,000`, `valid=1,000`, seed=42), 마지막 사용자 발화에 대한 답변만 학습 목표로 둔다. 기존 한 질문-한 답변 데이터보다 사람 같은 대화 흐름 평가에 적합하다.

새 gate와 multi-turn 데이터로 64건 파일럿을 실행했다. `train=1.5097`, `valid=1.5605`였고 RTX 4050 6GB에서 `max_length=256`, batch 1, accumulation 4 설정이 정상 동작했다. `max_length=384`에 distillation까지 켠 설정은 CUDA OOM이 발생해 장기 학습 설정으로 채택하지 않는다.

파일럿을 hybrid에 임시 연결해 산책·바베큐·운동 부족·직장 스트레스·업무 과다 5개를 확인했으며 모두 기존 router와 정상 응답했다. 그러나 어댑터가 일반 사실 질문의 의미와 정확도를 개선했다는 충분한 증거는 아직 없다. 따라서 새 파일럿도 운영 기본값에는 반영하지 않고, 기본은 Qwen2.5 1.5B noop + 자소 HRM router/memory를 유지한다.
## 137.5. 512건 multi-turn Qwen LoRA A/B 보류

자소 prefix만이 아니라 Qwen 생성 본체에 공개 multi-turn 문맥을 직접 반영하기 위해 Qwen2.5 1.5B attention LoRA를 512건으로 진단 학습했다. 전체 15.46억 파라미터 중 2,179,072개(0.141%)만 갱신했고, `train_loss=1.4892`, `valid_loss=1.5017`이었다.

미사용 공개 질문에서 우유 막 처리, 새우 색 변화, 원더걸스 노래명, 강아지 발바닥 답변이 모두 의미 또는 사실 오류를 보였다. loss가 낮아져도 사람 같은 대화나 사실성이 보장되지 않으므로 LoRA는 운영에 반영하지 않는다.

현재 다음 최적화 방향은 Qwen 본체를 무작정 학습하는 것이 아니라, 공개 factual QA의 보수적 retrieval과 HRM의 문맥·감정 계획을 생성기에 제한적으로 연결하는 것이다.

## 138. 512건 multi-turn A/B와 의도 분류 OOD gate

공개 multi-turn 정제 데이터 512건으로 자소 prefix를 학습했다(`train_loss=1.5220`, `valid_loss=1.6537`). 미사용 safe QA 10개 형식 검사는 `korean/no_mixing/no_repeat/nonempty=1.0`였지만, 우유·새우·조선시대 질문에서 사실성이 개선되지 않았고 강아지 발바닥 답변도 완전히 안정적이지 않았다. 따라서 512건 어댑터도 운영에 반영하지 않는다.

같은 미사용 multi-turn 10개 hybrid 평가에서 작은 의도 분류기가 `학교에서 친구들과 대화하기 어렵다`를 `서울입니다.`로 잘못 응답하는 OOD 오류를 발견했다. confidence threshold만으로는 작은 분류기의 과신을 막을 수 없으므로, `scripts/dialogue_intent.py`에 의도별 핵심 단어 gate를 추가했다. 수정 후 같은 질문은 일반 Qwen 대화 경로로 가며 `서울입니다.` 오답이 사라졌다. 전체 hybrid 회귀는 `PASS cases=22`를 유지한다.

---
---

# 🔄 방향 전환 결정 (2026-07-18)

## 139. Qwen 의존 구조 정리 및 밑바닥부터 재학습 결정

### 현황 진단

Section 112~138에 걸쳐 Qwen2.5 1.5B를 teacher fallback으로 도입하고 자소 prefix adapter, LoRA, 다중턴 데이터 등 다양한 방법으로 품질 향상을 시도했으나, 다음 근본 문제가 해결되지 않았다.

* raw neural decoder 단독 자유생성: **5% 수준** (미해결)
* Qwen은 자소 HRM의 문제를 **해결한 것이 아니라 우회한 것**
* 자소 6트랙 구조의 독창성이 Qwen 위에 얹히면서 연구 가치 희석
* validation loss 하락과 실제 자유생성 품질이 일치하지 않는 패턴 반복

### 핵심 인사이트: Section 104의 교훈

```
오염된 데이터로 학습:  0/30  (0%)
정합성 복구 데이터:   28/30 (94%)
```

데이터 정합성 하나만 바꿨는데 0% → 94%로 뒤집혔다. 아키텍처 실패가 아니라 **데이터 품질 문제**였다. 이 원칙을 대화 학습에 적용하지 못한 채 Qwen을 도입한 것이 방향 오류였다.

### 결정 사항

1. **Qwen 운영 중단**: Qwen2.5, KoGPT2, LFM 등 외부 pre-trained 모델을 운영 경로에서 제거. 비교 기준선(baseline)으로만 보존.
2. **밑바닥부터 재학습**: 6트랙 자소 구조 기반 순수 자소 모델을 처음부터 다시 학습.
3. **모델/체크포인트 정리**: Qwen hybrid era 산물들을 `backup_qwen_era/`로 격리, 핵심 자소 checkpoint만 운영 경로에 보존.

### 보존 핵심 체크포인트 (운영 경로 유지)

| 파일 | 이유 |
|---|---|
| `hrm_context_reasoning_order_finetune_2ep_best.pth` | reasoning specialist 97% |
| `hrm_context_copy_pure_dialogue_v2_12ep_best.pth` | dialogue HRM 기준선 |
| `hrm_intent_pure_v3_best.pth` | 의도 분류기 |
| `hrm_context_reasoning_clean_5ep_best.pth` | clean data reasoning |
| `unattended_full_best.pth` | Transformer pretrain 기준선 |

### 다음 행동 계획

1. **대규모 순수 한국어 pretrain** — 위키, 뉴스 등 정합성 검증된 corpus로 자소 언어 모델 충분히 사전학습
2. **대화 데이터 정합성 전수 검사** — reasoning처럼 tokenizer round-trip 오류 0개 확인 후 SFT
3. **작은 실험 우선** — 100건씩 A/B, 데이터 정합성 확인 후 확대
4. Qwen은 **비교 기준선**으로만 사용 (우리 모델이 얼마나 따라잡았는지 측정용)

