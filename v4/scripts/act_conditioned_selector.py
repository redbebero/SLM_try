"""Answer retrieval conditioned on a conservative Korean dialogue act."""

from collections import defaultdict

from conditional_selector import ConditionalSelector
from hangul_semantic_plan import classify_response_act


def label_dialogue_act(question):
    """Map lexical act labels to a small candidate-bank vocabulary."""
    return classify_response_act(question)


class ActConditionedSelector:
    def __init__(self, rows):
        self.rows = list(rows)
        self.groups = defaultdict(list)
        for row in self.rows:
            act = row.get("dialogue_act") or label_dialogue_act(row.get("question", ""))
            tagged = {**row, "dialogue_act": act}
            self.groups[act].append(tagged)
        self.all_selector = ConditionalSelector(self.rows)
        self.selectors = {act: ConditionalSelector(group)
                          for act, group in self.groups.items()}

    def select(self, question, *, exclude_pair_hash=None, min_score=0.0):
        act = label_dialogue_act(question)
        selector = self.selectors.get(act)
        result = selector.select(question, exclude_pair_hash=exclude_pair_hash,
                                 min_score=min_score) if selector else None
        if result is not None:
            return {**result, "dialogue_act": act}
        result = self.all_selector.select(question, exclude_pair_hash=exclude_pair_hash,
                                          min_score=min_score)
        return ({**result, "dialogue_act": act, "fallback": True}
                if result is not None else None)
