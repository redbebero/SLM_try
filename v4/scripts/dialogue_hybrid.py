"""Small plan-driven dialogue layer for the hybrid Korean model."""

import re

from hangul_semantic_plan import analyze_question, extract_semantic_slots, render_plan


TEMPLATES = {
    "인사": (
        "안녕하세요! 오늘은 어떤 이야기를 해볼까요?",
        "반가워요. 궁금한 점이나 하고 싶은 이야기를 말해 주세요.",
    ),
    "도움": (
        "네, 도와드릴게요. 무엇이 필요한가요?",
        "물론이에요. 문제를 조금 더 자세히 말해 주세요.",
    ),
    "위로": (
        "많이 힘들었겠어요. 무슨 일이 있었는지 말해도 괜찮아요.",
        "마음이 무겁고 지쳤군요. 지금 가장 힘든 부분은 무엇인가요?",
    ),
    "조언": (
        "작은 목표부터 정하고 하나씩 해보는 것이 좋아요.",
        "현재 상황을 나누어 본 뒤 가장 쉬운 일부터 시작해 보세요.",
    ),
    "되묻기": (
        "조금 더 구체적으로 말해 주면 같이 생각해 볼게요.",
        "질문의 대상이나 원하는 답변 형식을 알려 주세요.",
    ),
}


class DialogueManager:
    """Plan-first dialogue manager with bounded template candidates."""

    def __init__(self):
        self.memory = {}
        self.turns = []

    def _remember(self, text, slots):
        if "activity" in slots:
            self.memory["activity"] = slots["activity"]
        if "location" in slots:
            self.memory["location"] = slots["location"]
        if "item" in slots:
            self.memory["item"] = slots["item"]
        if "이름" in text:
            match = re.search(r"이름은\s*([가-힣]{2,8})", text)
            if match:
                self.memory["name"] = match.group(1)

    def _memory_reply(self, text):
        compact = re.sub(r"\s+", "", text)
        if any(word in compact for word in ("내이름", "이름이뭐")) and "name" in self.memory:
            return f"{self.memory['name']}님이라고 했어요."
        if any(word in compact for word in ("뭘한다고", "무엇을한다고", "뭐한다고")) and "activity" in self.memory:
            return f"주말마다 {self.memory['activity']}한다고 했어요."
        if any(word in compact for word in ("어디살", "사는도시", "사는지역")) and "location" in self.memory:
            return f"{self.memory['location']}에 산다고 했어요."
        return None

    def candidates(self, text):
        plan = analyze_question(text)
        slots = {**extract_semantic_slots(text), **plan.slots}
        self._remember(text, slots)
        memory = self._memory_reply(text)
        if memory:
            return [memory]
        if "activity" in slots and not any(mark in text for mark in ("?", "까", "나요", "어")):
            return [f"주말마다 {slots['activity']}하는군요. 기억해둘게요."]
        if "name" in self.memory and "이름" in text and not any(mark in text for mark in ("뭐", "무엇", "누구")):
            return [f"{self.memory['name']}님이라고 기억할게요."]
        specialist = None
        if plan.tool == "reasoning":
            specialist = render_plan(text, plan)
        if specialist and plan.intent in {"계산", "순서", "기억", "자모", "사실"}:
            return [specialist]
        if plan.intent == "인사":
            return list(TEMPLATES["인사"])
        if plan.intent == "위로":
            return list(TEMPLATES["위로"])
        if plan.intent == "조언":
            return list(TEMPLATES["조언"])
        if any(word in text for word in ("도와", "도움", "질문해")):
            return list(TEMPLATES["도움"])
        return list(TEMPLATES["되묻기"])

    def reply(self, text, candidate_index=0):
        candidates = self.candidates(text)
        answer = candidates[min(candidate_index, len(candidates) - 1)]
        self.turns.append({"input": text, "output": answer})
        return answer
