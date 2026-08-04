"""Compact dual encoder with a dialogue-act auxiliary objective."""

import torch
from torch import nn
import torch.nn.functional as F

from dual_selector import CompactDualEncoder


class ConditionedDualEncoder(CompactDualEncoder):
    def __init__(self, vocab_size, num_acts, emb_dim=32, hidden_dim=64,
                 output_dim=64, pad_id=0):
        super().__init__(vocab_size, emb_dim, hidden_dim, output_dim, pad_id)
        self.act_head = nn.Linear(output_dim, num_acts)

    def forward_with_act(self, question_ids, answer_ids):
        question = self.encode(question_ids)
        answer = self.encode(answer_ids)
        return question, answer, self.act_head(question), self.act_head(answer)

    def act_loss(self, question_ids, answer_ids, labels):
        _, _, question_logits, answer_logits = self.forward_with_act(
            question_ids, answer_ids)
        return (F.cross_entropy(question_logits, labels)
                + F.cross_entropy(answer_logits, labels)) * 0.5
