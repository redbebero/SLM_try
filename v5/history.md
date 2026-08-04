# SLM v5 기록

작은 모델이 한국어 영창을 제한된 마법 효과로 분류하고, 결정론적 주문 엔진이 이를 누적·검증·발동하는 브라우저 마법 RPG를 만든다.

---

## 확정된 방향

- 브라우저 우선으로 개발한다.
- 초기 기술은 HTML, CSS, TypeScript, Canvas, IndexedDB다.
- 사용자는 한국어 영창을 입력한다.
- AI는 단어 분류와 후보 효과 제안만 한다.
- 주문·전투 엔진이 마나, 피해, 안정성, 랜덤, 상태 변경을 담당한다.
- 원소·형태·위력·속도·거리·지속시간을 주문 상태로 누적한다.
- 사건 로그와 seed로 전투를 재현한다.
- AI가 없어도 고정 분류 fixture로 주문 엔진을 검증할 수 있어야 한다. 한국어 단어 매핑 사전은 만들지 않는다.
- PyTorch 모델을 먼저 만들고 ONNX Runtime Web으로 브라우저에 배포한다.
- Godot은 브라우저 프로토타입 검증 후 필요한 경우에만 검토한다.

---

## 현재 게임 설계

- 장르: 작은 마법 RPG 전투
- 초기 화면: 1개 전투 장소, 플레이어 1명, 적 1종
- 초기 원소: FIRE, WATER, AIR
- 초기 형태: ORB, SPEAR, SHIELD
- 초기 효과: POWER_UP, SPEED_UP, RANGE_UP, DURATION_UP
- 주문 단계: ELEMENT → FORM → MODIFIER → CAST
- 렌더링: Canvas 전투 화면과 HTML/CSS 상태·로그 화면

## 보류한 방향

- Luma 2D 대형 세계
- 통제 영어·Toki Pona·통제 한국어 비교 모델
- 자유로운 세계 행동과 NPC·퀘스트
- 세계 전체를 이해하고 설명하는 모델

위 방향들은 아이디어로 보존하지만 현재 v5 구현 범위에는 포함하지 않는다.

---

## 기록 규칙

- 진행 중인 작업과 미완료 결정은 `plan.md`에만 둔다.
- 계획이 완료되면 `plan.md`에서 삭제하고 이 파일에 결과를 기록한다.
- 실패·폐기·변경도 원인과 함께 이 파일에 기록한다.
- 코드 변경 전후에는 v5 저장소(`/home/redbebero/Projects/SLM/v5`) 전용 `code-review-graph`를 사용한다.
- 작업 시작 시 최소 컨텍스트와 위험도를 확인하고, 변경 후 그래프를 증분 갱신한다.
- 코드 리뷰 시 변경 영향 범위·영향 흐름·테스트 공백을 우선 확인한다.
- 모델·데이터셋·체크포인트가 실제로 존재하기 전에는 성공으로 기록하지 않는다.
- 손실 감소만으로 성공 판정하지 않는다. 상태 추적, 규칙 적용, 새로운 조합, 장기 일관성을 함께 확인한다.

---

## 현재 상태

### [2026-07-28] v5를 AI 전용 저장소로 범위 확정

- 상태: `completed`
- v5에서는 Godot, 브라우저, 전투 엔진, 세계 상태, UI를 구현하지 않는다.
- v5의 산출물은 학습 데이터, 전처리 규칙, 학습된 모델, 평가 결과, ONNX handoff package다.
- 기존 토큰 단위 `ClassifiedToken` 계획을 폐기하고, 전체 한국어 영창을 읽는 `SpellProposal` 모델로 전환한다.
- AI는 원소·형태·대상·위력·속도·거리·지속시간을 제안한다.
- 마나·정신력·피해·HP 등 게임 상태는 모델 출력에 넣지 않는다. 별도 Godot 엔진이 최종 계산한다.
- 설계 문서: `docs/superpowers/specs/2026-07-28-ai-only-spell-model-design.md`
- 다음 작업: `SpellProposal` Schema·타입·실패 테스트 작성.

### [2026-07-28] AI 전용 학습·ONNX 수직 슬라이스 구현

- 상태: `completed`
- 전체 영창 입력용 `SpellProposal` Schema·TypeScript 타입·검증 함수를 추가했다.
- 학습 레코드는 토큰이 아니라 전체 영창과 `split_group`을 저장하도록 추가했다.
- UTF-8 byte encoder를 구현했다. 각 byte에 1을 더하고, 0을 padding으로 사용하며, 길이는 96으로 고정한다. 별도 tokenizer 파일이 필요 없다.
- 작은 multi-task PyTorch 모델을 구현했다. 출력 head는 `status`, `element`, `form`, `target`, `power`, `speed`, `range`, `duration`이다.
- 학습·평가·ONNX export 명령을 추가했다. ONNX 입력은 `input_ids`, 출력 이름과 순서는 handoff 문서에 고정했다.
- CPU 전용 PyTorch 환경을 Python 3.12로 고정했다. CUDA 패키지는 제외했다.
- 초기 데이터는 20개 레코드뿐이다. 16개 train, 2개 dev, 2개 test로 분할했다.
- 검증 결과: TypeScript 17개 통과, Python 15개 통과, typecheck 통과, ONNX Runtime 입출력 및 PyTorch 출력 일치 테스트 통과.
- 초기 모델 결과: train complete accuracy 1.0, dev 0.5, test 0.0. 데이터가 너무 작아 모델 성공으로 판정하지 않는다.
- 산출물: `models/spell_ai.pt` checkpoint, `models/spell_ai.onnx` 95KB. Godot 코드는 추가하지 않았다.
- ONNX Runtime 실제 추론 확인: `붉은 불꽃 구체를 적에게 날려`는 FIRE/ORB/ENEMY proposal을 반환했고, 일반 문장 `오늘 저녁에는 책을 읽고 싶다`는 UNKNOWN을 반환했다. 이는 seed 데이터 암기 확인이며 일반화 증거가 아니다.
- code-review-graph full rebuild 결과: 27 files, 90 nodes, 670 edges, 3 flows. 변경 위험도 `0.65`; 미테스트로 표시된 CLI/forward 경로는 다음 테스트 보강 대상이다.
- 다음 작업: 학습 데이터 증량과 schema 기반 JSONL 검증 명령.

### [2026-07-29] 데이터 증량·검증 및 기준 모델 재학습

- 상태: `partial`
- `data/seed-spells.jsonl`을 추가했다. 사람 작성 18개, 공개도메인 Oz 각색 9개, hard negative 10개를 추가해 base record가 57개가 됐다.
- 전체 base 구성은 사람 작성 30개, 공개도메인 각색 10개, hard negative 17개다. 각 변형은 같은 `split_group`을 유지한다.
- `training.generate_dataset`가 기본 문구에 통제된 시전 표현 변형을 붙여 422개 레코드를 만들었다. 구성은 PROPOSAL 320개, UNKNOWN 102개다.
- `training.validate_dataset`를 추가하고 생성 데이터 전체가 `training-spell-record.schema.json`을 통과하는 것을 확인했다.
- 기존 데이터의 `LIGHT` 원소가 schema·label table에서 빠져 있던 계약 불일치를 수정했다. 모델 element head는 7개에서 8개로 변경했다. 이는 단어 사전 추가가 아니라 공유 출력 계약 수정이다.
- train/evaluate 기본 입력을 `data/training-spells.expanded.jsonl`로 변경했다.
- 재학습 결과: train complete accuracy `1.0`, dev `0.1579`, test `0.1579`; 손실 감소만으로 성공 판정하지 않는다.
- `models/spell_ai.pt`와 `models/spell_ai.onnx`를 새 출력 계약으로 재생성했다. 다음 작업은 unknown detection·per-class metric·baseline 비교다.
- 검증 결과: TypeScript 18개 통과, Python 19개 통과, typecheck 통과. ONNX Runtime 출력은 `input_ids [batch,96]`, element 8개 head를 포함한 고정 8개 output이다.
- 변경 후 code-review-graph full rebuild 결과: 31 files, 104 nodes, 787 edges, 5 flows, 5 communities. 변경 분석 위험도는 `0.85`이며 CLI·학습 경로 직접 테스트 공백이 10개로 남았다. 다음 평가 단계에서 보강한다.

### [2026-07-29] 평가 지표·majority baseline 추가

- 상태: `completed`
- `training.evaluate`에 UNKNOWN binary detection accuracy/precision/recall/F1/false-positive-rate를 추가했다.
- `status`, `element`, `form`, `target`에 대해 per-class support/precision/recall/F1을 추가했다.
- train split에서만 계산한 majority-label baseline을 dev/test와 비교하도록 CLI를 확장했다. 평가 split의 정답을 baseline 생성에 사용하지 않는다.
- 새 지표로 확인한 test 결과: model UNKNOWN recall `1.0`, F1 `0.571`; majority baseline UNKNOWN recall `0.0`. model complete accuracy `0.1579`, baseline `0.0`이다.
- 모델은 여전히 일반화 성능이 낮으므로 release handoff 대상이 아니다. 다음 작업은 byte model과 작은 한국어 encoder 비교다.
- 검증 결과: TypeScript 18개 통과, Python 21개 통과, typecheck 통과. 변경 후 code-review-graph는 31 files, 113 nodes, 856 edges, 5 flows를 기록했고 현재 변경 위험도는 `0.65`다.

### [2026-07-29] 외부 애니·게임 영창 직접 추론 검증

- 상태: `failed-generalization`
- 학습에 사용하지 않는 `data/external-incantation-eval.jsonl`을 추가했다. 짧은 출처 귀속 입력 6개만 저장하고, 원문 전체나 정답 label은 저장하지 않았다.
- Fate UBW 한국어 영창의 짧은 입력 3개와 Elden Ring 한국어 기도명·영창 3개를 현재 ONNX로 추론했다.
- 결과: 6개 중 5개는 `UNKNOWN`이었지만 confidence가 `0.995~0.999`로 과도하게 높았다. 1개는 `EARTH/BEAM/AREA`, confidence `0.460`인 잘못된 proposal이었다.
- 전체 UBW 영창은 저장하지 않고 일회성으로 추론했다. 결과는 `PROPOSAL / SHADOW / SPEAR / ENEMY`, confidence `0.785`로 잘못 분류됐다.
- 결론: 현재 모델은 훈련 문구 밖의 실제 작품 영창을 이해하지 못한다. confidence도 불확실성으로 믿을 수 없다. 외부 평가 입력은 training dataset에 병합하지 않는다.

### [2026-07-29] 공백 단위 다중 속성 토큰 모델 구현

- 상태: `partial`
- AI 입력을 전체 영창에서 공백 기준 토큰 하나로 축소했다. 문장 문맥·마나·HP·피해·위치·인벤토리·NPC 상태는 입력과 출력에서 제외했다.
- 토큰 출력은 `attributes[]` 다중 라벨로 고정했다. 각 속성은 `kind`, `value`, `delta`, `confidence`를 가진다. Godot은 이 배열을 조합하고 주문 규칙을 실행한다.
- `ELEMENT`, `FORM`, `TARGET`, `INTENT`, `MODIFIER`, `SIZE`, `DIRECTION`, `QUANTITY`, `CAST` 9개 kind와 총 82개 영어 atomic label을 계약으로 추가했다.
- 한국어 표현은 코드 사전이 아니라 `data/token-lexicon.json`에 넣었다. 86개 의미 entry, 434개 1토큰 표면형을 만들었다. `붉은`, `빨간`, `붉다`, `화염`, `불꽃`은 모두 `ELEMENT:FIRE`다. ordinary/hard-negative 단어도 포함했다.
- `training.generate_token_dataset`가 lexicon을 JSONL로 생성하고, `training.validate_token_dataset`가 strict schema를 검증한다. 생성 데이터 434개가 검증을 통과했다.
- 새 모델 산출물: `models/token_ai.pt` 146,261 bytes, `models/token_ai.onnx` 144,185 bytes. ONNX 출력은 `attribute_logits`, `delta_logits` 두 개이며 atomic label 순서는 checkpoint에 저장했다.
- 직접 추론 결과: `붉은`, `빨간`, `불꽃`, `화염` → FIRE; `구체` → ORB; `적에게` → ENEMY; `빠르게` → SPEED_UP; `오래` → DURATION_UP; `모두` → AREA+ALL; 등록한 hard negative는 UNKNOWN.
- ONNX와 PyTorch logits parity를 실제 16개 토큰에서 확인했다.
- 실패/한계: byte 모델은 표면형을 암기하는 데는 강하지만 새 동의어 일반화가 약하다. held-out surface-form exact accuracy는 dev `0.103`, test `0.149`였고 false positive도 발생했다. 학습 loss `0.00194`만으로 성공 판정하지 않는다.
- 변경 후 code-review-graph full rebuild 결과: 48 files, 175 nodes, 1,317 edges, 8 flows, 5 communities. 변경 위험도 `0.85`, 테스트 공백 11개가 남았다. 핵심 공백은 token training CLI·forward 경로·lexicon I/O다.
- 다음 작업: 표면형 분할·라벨 지원량을 유지한 평가 개선, UNKNOWN threshold/calibration, 소형 한국어 pretrained encoder 비교.

### [2026-07-29] 한국어 토큰 분류 터미널 CLI 추가

- 상태: `completed`
- `uv run python -m training.token_cli "한국어 영창"` 명령으로 실제 한국어 문장을 검증할 수 있게 했다.
- 입력이 없으면 stdin을 읽는다. `--json`은 전체 결과 JSON을 출력하고, 기본 출력은 토큰·속성·delta·confidence를 사람이 읽기 쉽게 표시한다.
- 공백 기준 토큰 분리와 토큰별 독립 분류를 고정했다. AI는 속성 조합, 마나, 피해, 게임 상태를 계산하지 않는다.
- 실제 확인: `주변 적에게 붉은 불꽃 구체 빠르게 날려`에서 AREA, ENEMY, FIRE, ORB, SPEED_UP, MOVE를 출력했다. 조사 결합형 `창을`은 현재 UNKNOWN으로 표시되어 데이터 보강 대상이다.
- 변경 파일: `training/token_cli.py`, `tests/training/test_token_cli.py`, `training/README.md`, `models/README.md`.
- 검증: CLI 전용 테스트 5개와 전체 Python 테스트 40개가 통과했다. ONNX deprecation warning 2개는 기존 경고이며 실패가 아니다.
- 변경 후 code-review-graph full rebuild 결과: 50 files, 190 nodes, 1,424 edges, 9 flows, 5 communities. 변경 분석은 테스트 공백 2개(`load_token_checkpoint`, 테스트 고정 모델의 `__call__`)를 남겼다.

### [2026-07-25] v5 목표 재정의

- 상태: `completed`
- 기존 목표인 일반 한국어 모델 학습은 중단한다.
- 새 목표는 작은 언어와 작은 판타지 세계를 통합한 텍스트 게임 모델이다.
- 첫 언어는 통제 영어로 정했다.
- Toki Pona, 확장 Toki Pona, 통제 한국어는 동일 조건의 비교 대상으로 남긴다.
- 다음 작업: 2D 세계 상태와 최소 규칙 정의

### [2026-07-25] code-review-graph 관리 기반 설정

- 상태: `completed`
- 상위 Git 저장소를 기준으로 그래프를 초기화했다.
- 현재 그래프 요약: 123개 노드, 2276개 엣지, 위험도 `low (0.40)`.
- v5에는 아직 코드가 없으므로 현재 분석 결과는 주로 v4 변경 코드에 해당한다.
- 이후 v5 코드가 추가되면 증분 갱신 후 변경 영향·흐름·테스트 공백을 확인한다.

### [2026-07-25] v5 저장소·그래프 분리

- 상태: `completed`
- v5에 독립 Git 저장소를 생성했다.
- v5 전용 `code-review-graph`를 새로 초기화했다.
- 초기 그래프는 코드가 없어 0개 파일, 0개 노드, 0개 엣지다.
- v1~v4의 코드와 그래프는 보존하며 v5 작업 분석에는 포함하지 않는다.

### [2026-07-25] 영창 마법 RPG 계획 구체화

- 상태: `completed`
- `plan.md`를 영창 입력 → 단어 분류 → 주문 상태 누적 → 결정론적 발동 계산 흐름으로 전면 교체했다.
- AI는 토큰 분류와 후보 효과만 제안하고, 엔진이 phase·자원·피해·랜덤·상태 변경을 담당하도록 경계를 명시했다.
- 고정 JSON 기준선 → 주문 엔진 → 전투·로그 → 브라우저 Canvas → 제한 라벨 분류기 → 소형 모델 → ONNX → 데이터 확장 순서를 확정했다.
- Godot은 초기 구현에서 제외하고, 브라우저 프로토타입 검증 후 필요할 때만 검토한다.

### [2026-07-25] 실행 큐와 그래프 운영 절차 정리

- 상태: `completed`
- 첫 구현은 AI·Canvas·IndexedDB 없이 계약 타입, JSON Schema, 고정 fixture부터 시작하도록 우선순위를 정했다.
- v5 전용 `code-review-graph`를 작업 전 최소 컨텍스트, 작업 후 증분 갱신, 변경 영향·흐름·테스트 공백 확인에 사용하도록 절차를 추가했다.
- 상위 저장소 그래프를 사용한다는 기존 기록을 v5 독립 그래프 기준으로 수정했다.

### [2026-07-25] 파인튜닝·공개도메인 fixture 전략 결정

- 상태: `completed`
- 웹 게임 배포 방식은 처음부터 언어 모델을 학습하지 않고, 작은 한국어 사전학습 인코더를 제한 라벨 분류기로 파인튜닝하는 방향으로 정했다.
- 후보는 MIT 라이선스가 표시된 `beomi/KcELECTRA-small-v2022`이며, 최종 ONNX 크기와 브라우저 지연을 측정한 뒤 확정한다.
- 판타지 문구는 저작권이 확인된 상용 게임·소설에서 복사하지 않고, Wikisource의 공개도메인 Oz 작품을 출처가 붙은 시험 fixture로 사용한다.
- 실제 게임에서는 단어 사전을 코드에 넣지 않고 `토큰 + 주문 문맥 → 제한 라벨` 모델 출력만 사용한다. 엔진은 출력 라벨의 허용 여부와 효과 계산만 담당한다.

### [2026-07-25] 계약·출처 fixture 초기 구현

- 상태: `completed`
- TypeScript, Vitest, Ajv 기반 최소 프로젝트 설정을 추가했다.
- AI 출력의 허용 라벨과 `hp`·`mana`·`damage` 직접 변경 금지를 JSON Schema와 테스트로 고정했다.
- Baum의 Oz 작품에서 출처 URL이 있는 짧은 판타지 문구 3개를 fixture로 추가했다. fixture는 단어 매핑 사전이 아니다.
- 검증 결과: 테스트 5개 통과, TypeScript 타입 검사 통과.
- 그래프 결과: 5개 파일, 13개 노드, 85개 엣지. 고립 노드 0개, 미테스트 hotspot 0개. 분류기와 주문 엔진은 다음 작업 대상이다.

### [2026-07-25] 한국어 영창 학습 데이터 생성 절차 확정

- 상태: `completed`
- 학습 목표를 한국어 영창 생성이 아니라 `토큰 + 읽기 전용 주문 문맥 → 제한 라벨` 분류로 고정했다.
- 레코드는 전체 영창, 현재 토큰, prefix, 토큰 위치, `spell_context`, 정답 라벨, 출처·검수 정보를 함께 가진다.
- 데이터는 사람 작성 영창, 출처 URL이 있는 공개도메인 각색, 통제된 변형, hard negative 순서로 생성한다.
- train/dev/test는 토큰 단위 무작위 분할이 아니라 전체 영창·템플릿·출처 단위로 분할해 표현 누수를 막는다.
- 첫 목표량은 사람 작성 30개, 공개도메인 각색 10개, 변형 약 300개, hard negative 약 100개다.
- 이번 단계에서는 데이터셋·체크포인트를 만들지 않았다. 다음 승인 대상은 분류기 인터페이스와 고정 분류기 테스트다.

### [2026-07-25] 학습 데이터 레코드 Schema 구현

- 상태: `completed`
- `input`, `target`, `provenance`를 분리한 `training-token-record.schema.json`을 추가했다.
- 주문 문맥은 읽기 전용 `phase`, `element`, `form`, `power`, `speed`만 허용한다.
- 공개도메인 각색 레코드는 `source_url`을 필수로 하고, 모든 객체의 추가 필드를 거부한다.
- `hp`, `mana`, `damage`, 미지원 라벨을 거부하는 테스트를 포함했다.
- 검증 결과: 전체 테스트 10개 통과, TypeScript 타입 검사 통과.
- 그래프 결과: 전체 재구축 `6 files, 20 nodes, 132 edges`, 위험도 `0.50`, 테스트 공백 `0`.
