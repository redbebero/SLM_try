"""Small hierarchical jamo encoder for dialogue-intent classification."""

import torch
import torch.nn as nn


class HRMIntentNet(nn.Module):
    def __init__(self, vocab_sizes, num_classes, emb_dim=16, hidden_dim=96,
                 cycle_steps=4, max_seq_length=160):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cycle_steps = cycle_steps
        self.max_seq_length = max_seq_length
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, emb_dim, padding_idx=0) for size in vocab_sizes
        ])
        self.type_emb = nn.Embedding(4, emb_dim)
        self.input_proj = nn.Linear(emb_dim * 7, hidden_dim)
        self.l_cell = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)
        self.h_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.l_norm = nn.LayerNorm(hidden_dim)
        self.h_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    @staticmethod
    def _types(x):
        return ((x[:, :, 3] > 0).long()
                + 2 * (x[:, :, 4] > 0).long()
                + 3 * (x[:, :, 5] > 0).long())

    def forward(self, x, mask=None):
        types = self._types(x)
        previous_jong = torch.cat([
            torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device),
            x[:, :-1, 2],
        ], dim=1)
        tracks = [x[:, :, 0], x[:, :, 1], previous_jong,
                  x[:, :, 3], x[:, :, 4], x[:, :, 5]]
        encoded = [emb(token) for emb, token in zip(self.embeddings, tracks)]
        encoded.append(self.type_emb(types))
        inputs = self.input_proj(torch.cat(encoded, dim=-1))
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        l = torch.zeros_like(h)
        states = []
        for index in range(inputs.size(1)):
            l = self.l_norm(self.l_cell(torch.cat([inputs[:, index], h], dim=-1), l))
            if (index + 1) % self.cycle_steps == 0:
                h = self.h_norm(self.h_cell(l, h))
            states.append(torch.cat([h, l], dim=-1))
        states = torch.stack(states, dim=1)
        if mask is None:
            pooled = states[:, -1]
        else:
            weights = mask.to(states.dtype).unsqueeze(-1)
            pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def load_intent_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    vocab_sizes = [20, 22, 28, 36, 53, 11]
    for index in range(len(vocab_sizes)):
        weight = checkpoint["model"].get(f"embeddings.{index}.weight")
        if weight is not None and weight.ndim == 2:
            vocab_sizes[index] = weight.shape[0]
    model = HRMIntentNet(
        vocab_sizes=tuple(vocab_sizes),
        num_classes=len(checkpoint["labels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.intent_labels = checkpoint["labels"]
    model.intent_responses = checkpoint["responses"]
    return model


def predict_intent(model, tokenizer, question, threshold=0.75):
    """Return a conservative learned reply for a supported intent."""
    if "Q:" in question and "A:" in question:
        question = question.rsplit("Q:", 1)[-1].split("A:", 1)[0].strip()
    if model is None or any(char.isdigit() for char in question):
        return None
    if any(word in question for word in ("지문:", "질문:", "사는 곳", "거주지", "산 것은")):
        return None
    x = tokenizer.encode(question).unsqueeze(0).to(next(model.parameters()).device)
    mask = torch.ones(1, x.size(1), device=x.device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x, mask), dim=-1)[0]
    score, index = probabilities.max(dim=-1)
    if float(score) < threshold:
        return None
    label = model.intent_labels[int(index)]
    if label.startswith("context_") or label == "unknown":
        return None
    # The tiny intent model is intentionally overconfident on out-of-domain
    # Korean.  Require an explicit lexical cue before allowing a canned fact
    # or advice answer to replace the generative dialogue path.
    cue_groups = {
        "capital": ("수도", "대한민국", "우리나라"),
        "king": ("한글", "훈민정음", "세종", "왕", "임금"),
        "cat": ("고양이", "동물"),
        "python": ("파이썬", "python"),
        "moon": ("달", "위성"),
        "greet_help": ("안녕", "도와"),
        "greet_welcome": ("반가", "처음"),
        "greet_well": ("잘 지냈", "기분"),
        "advice_ask": ("궁금", "물어", "질문"),
        "advice_study": ("공부", "학습", "시험", "연습"),
        "advice_mistake": ("실수", "틀린", "잘못", "다시"),
        "advice_tasks": ("할 일", "해야 할 일", "일정", "정리"),
    }
    cues = cue_groups.get(label)
    if cues is None or not any(cue in question for cue in cues):
        return None
    return model.intent_responses[label]
