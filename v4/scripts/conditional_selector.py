"""Select a downloaded answer candidate without generating new training text."""

import re
import unicodedata
from collections import defaultdict

from retrieval import TfidfCharNgramRetriever


def normalize_question(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


class ConditionalSelector:
    """Exact-first, train-only fuzzy answer selector."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.retriever = TfidfCharNgramRetriever(self.rows)
        self.exact = defaultdict(list)
        for row in self.rows:
            self.exact[normalize_question(row.get("question", ""))].append(row)

    def select(self, question, *, exclude_pair_hash=None, min_score=0.0):
        key = normalize_question(question)
        exact_rows = [
            row for row in self.exact.get(key, [])
            if row.get("pair_hash") != exclude_pair_hash
        ]
        if exact_rows:
            row = exact_rows[0]
            return {**row, "score": 1.0, "method": "exact"}

        hits = self.retriever.search(question, top_k=len(self.rows))
        for hit in hits:
            if hit.get("pair_hash") == exclude_pair_hash:
                continue
            if hit["score"] >= min_score:
                return {**hit, "method": "fuzzy"}
        return None
