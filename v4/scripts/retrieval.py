"""Dependency-free character n-gram retrieval for a compact Korean assistant."""

import math
from collections import Counter


def _grams(text, n=3):
    text = "".join(text.lower().split())
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def _jaccard(left, right):
    shared = sum(min(left[key], right[key]) for key in left.keys() & right.keys())
    total = sum(left.values()) + sum(right.values()) - shared
    return shared / total if total else 0.0


class CharNgramRetriever:
    def __init__(self, rows, n=3):
        self.rows = list(rows)
        self.n = n
        self.index = [_grams(row.get("question", ""), n) for row in self.rows]

    def search(self, question, top_k=3):
        query = _grams(question, self.n)
        scored = [(_jaccard(query, grams), row) for grams, row in zip(self.index, self.rows)]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**row, "score": score} for score, row in scored[:top_k]]


class TfidfCharNgramRetriever:
    """Character retrieval that downweights n-grams shared by many questions."""

    def __init__(self, rows, n=3):
        self.rows = list(rows)
        self.n = n
        self.index = [_grams(row.get("question", ""), n) for row in self.rows]
        document_frequency = Counter()
        for grams in self.index:
            document_frequency.update(grams.keys())
        count = max(1, len(self.rows))
        self.idf = {gram: math.log((count + 1) / (frequency + 1)) + 1.0 for gram, frequency in document_frequency.items()}
        self.vectors = [self._vector(grams) for grams in self.index]
        self.postings = {}
        for row_id, grams in enumerate(self.index):
            for gram in grams:
                self.postings.setdefault(gram, set()).add(row_id)

    def _vector(self, grams):
        vector = {gram: (1.0 + math.log(count)) * self.idf.get(gram, 1.0) for gram, count in grams.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {gram: value / norm for gram, value in vector.items()} if norm else {}

    def search(self, question, top_k=3, source=None, category=None):
        query = self._vector(_grams(question, self.n))
        candidate_ids = set()
        query_grams = sorted(query, key=lambda gram: self.idf.get(gram, 0.0), reverse=True)[:12]
        for gram in query_grams:
            candidate_ids.update(self.postings.get(gram, set()))
        candidates = []
        for row_id in candidate_ids:
            vector, row = self.vectors[row_id], self.rows[row_id]
            if source is not None and row.get("source") != source:
                continue
            if category is not None and row.get("category") != category:
                continue
            score = sum(query.get(gram, 0.0) * value for gram, value in vector.items())
            candidates.append((score, row))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [{**row, "score": score} for score, row in candidates[:top_k]]
