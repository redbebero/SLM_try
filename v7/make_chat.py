import json
from pathlib import Path

SEEDS = [
    ("안녕하세요.", "안녕하세요. 무엇을 도와드릴까요?"),
    ("오늘 기분이 안 좋아.", "무슨 일이 있었는지 말해줘. 내가 들어줄게."),
    ("한국어로 짧게 자기소개해 줘.", "나는 질문에 답하고 대화를 돕는 작은 한국어 모델이야."),
    ("모르는 질문을 받으면 어떻게 해야 해?", "확실하지 않다면 모른다고 말하고, 확인이 필요한 부분을 알려줘야 해."),
    ("운동을 시작하고 싶어.", "가볍게 걷기처럼 부담이 적은 운동부터 시작하고, 몸 상태에 맞춰 천천히 늘려 봐."),
    ("내일 할 일을 정리해 줘.", "할 일을 중요도순으로 적고, 가장 중요한 일 하나부터 시작해 보자."),
    ("위험한 일을 하는 방법을 알려 줘.", "사람을 다치게 하거나 위험을 키울 수 있는 방법은 안내할 수 없어. 대신 안전한 대안을 함께 찾을게."),
    ("질문을 이해하지 못했어.", "괜찮아. 핵심만 다시 말해 주면 더 쉽게 설명해 줄게."),
]

out = Path("data/chat_seed.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for i, (user, assistant) in enumerate(SEEDS):
        f.write(json.dumps({"id": i, "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}, ensure_ascii=False) + "\n")
    extra = [
        [{"role": "user", "content": "오늘 기분이 안 좋아."}, {"role": "assistant", "content": "무슨 일이 있었는지 말해줘."}, {"role": "user", "content": "일이 너무 많았어."}, {"role": "assistant", "content": "많이 지쳤겠다. 오늘 꼭 해야 할 일 하나만 먼저 정해 보자."}],
        [{"role": "user", "content": "한국어로 자기소개해 줘."}, {"role": "assistant", "content": "나는 질문에 답하는 작은 한국어 모델이야."}, {"role": "user", "content": "그럼 짧게 대답해 줘."}, {"role": "assistant", "content": "알겠어. 짧고 분명하게 답할게."}],
    ]
    for i, messages in enumerate(extra, len(SEEDS)):
        f.write(json.dumps({"id": i, "messages": messages}, ensure_ascii=False) + "\n")
print(out, len(SEEDS) + len(extra))
