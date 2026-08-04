"""Conservative extractive fallback for Korean passage QA."""

import re

_STOPWORDS = set(
    "지문 질문 무엇 누구 언제 어디 어느 몇 어떤 왜 어떻게 은 는 이 가 을 를 의 "
    "에 에서 으로 로 와 과 한 알려 말해 주세요 까요 인가요 인가".split()
)


def _terms(text):
    text = re.sub(r"[^0-9A-Za-z가-힣 ]+", " ", text.lower())
    terms = set()
    for term in text.split():
        if len(term) > 1 and term not in _STOPWORDS:
            stem = term.rstrip("은는이가을를의에한")
            if len(stem) > 1:
                terms.add(stem)
    return terms


def extract_passage_answer(prompt, min_overlap=1):
    """Return the best source sentence, or None when confidence is low."""
    match = re.search(r"지문\s*:\s*(.*?)\s*질문\s*:\s*(.*)", prompt, re.S)
    if not match:
        return None
    passage, question = match.groups()
    question_terms = _terms(question)
    if not question_terms:
        return None
    # Do not split decimal points such as ``pH = 5.8``.
    sentences = [part.strip() for part in re.split(r"(?<=[!?。])\s*|(?<=\.)\s+(?=[가-힣A-Za-z《])", passage) if part.strip()]
    candidates = []
    for index, sentence in enumerate(sentences):
        overlap = len(question_terms & _terms(sentence))
        if overlap >= min_overlap:
            candidates.append((overlap, -len(sentence), -index, sentence))
    return max(candidates)[-1] if candidates else None
