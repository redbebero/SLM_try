"""Tiny lexical QA memory for a small HRM hybrid system."""

import re
from pathlib import Path


def _terms(text):
    # Small Korean paraphrase map keeps memory useful without a language model.
    text = text.lower()
    for source, target in (
        ("대한민국", "한국"), ("우리나라", "한국"), ("수도는", "수도"),
        ("알려줘", "알려"), ("말해줘", "알려"), ("창제한", "만든"),
        ("창제", "만든"), ("임금", "왕"), ("누구인가요", "누구"),
        ("무엇인가요", "무엇"), ("어떤 종류의", "어떤"),
    ):
        text = text.replace(source, target)
    text = re.sub(r"[^0-9A-Za-z가-힣]+", " ", text)
    terms = {term for term in text.split() if len(term) > 1}
    stems = {
        term.rstrip("은는이가을를의에")
        for term in terms
        if len(term.rstrip("은는이가을를의에")) > 1
    }
    return terms | stems


def _ngrams(text, width=2):
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i:i + width] for i in range(max(0, len(compact) - width + 1))}


class KnowledgeMemory:
    """Memory with conservative lexical retrieval and no model parameters."""

    def __init__(self, pairs=()):
        self.records = []
        for question, answer in pairs:
            self.add(question, answer)

    def add(self, question, answer):
        self.records.append((question.strip(), answer.strip(),
                             _terms(question), _ngrams(question)))

    def retrieve(self, query, threshold=0.42):
        # Advice/open-ended questions need generation, not nearest-answer copy.
        if re.search(r"어떻게|해야|방법|할까요|태도|정리하고 싶|물어봐|궁금한|될까요", query):
            return None
        query_terms = _terms(query)
        query_ngrams = _ngrams(query)
        if not query_terms or not query_ngrams:
            return None
        best = None
        for question, answer, terms, ngrams in self.records:
            shared_terms = query_terms & terms
            generic = {"무엇", "무엇인가요", "누구", "누구인가요", "위성은",
                       "언어인가요", "알려", "알려줘"}
            content_shared = shared_terms - generic
            if len(content_shared) < 2:
                continue
            category_terms = ("대통령", "수도", "위성", "언어", "포유류", "왕")
            if any(term in query_terms for term in category_terms):
                if any(term in query_terms and term not in terms
                       for term in category_terms):
                    continue
            term_score = len(query_terms & terms) / max(1, len(query_terms | terms))
            ngram_score = len(query_ngrams & ngrams) / max(1, len(query_ngrams | ngrams))
            score = 0.65 * term_score + 0.35 * ngram_score
            if best is None or score > best[0]:
                best = (score, question, answer)
        if best is None or best[0] < threshold:
            return None
        return {"score": best[0], "question": best[1], "answer": best[2]}


def load_sft_memory(paths):
    memory = KnowledgeMemory()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for block in re.split(r"\n\s*\n", text):
            q = re.search(r"(?m)^Q:\s*(.+)$", block)
            a = re.search(r"(?m)^A:\s*(.+)$", block)
            if q and a:
                memory.add(q.group(1), a.group(1))
    return memory
