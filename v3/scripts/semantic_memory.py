"""Conservative multilingual-E5 retrieval for public factual QA memory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SemanticMemory:
    def __init__(self, index_path, model_path, device="cuda", threshold=0.90, margin=0.02):
        self.index_path = Path(index_path)
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vectors = np.load(self.index_path.with_suffix(".npy"), mmap_mode="r")
        self.records = json.loads(self.index_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        self.threshold = threshold
        self.margin = margin

    def retrieve(self, query):
        if "Q:" in query:
            query = query.rsplit("Q:", 1)[-1]
            if "A:" in query:
                query = query.split("A:", 1)[0]
        query = query.strip()
        batch = self.tokenizer(
            ["query: " + query], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vector = torch.nn.functional.normalize(vector, p=2, dim=1).float().cpu().numpy()[0]
        scores = np.asarray(self.vectors, dtype=np.float32) @ vector
        best = int(scores.argmax())
        top = np.partition(scores, -2)[-2:]
        second = float(top[0] if top[1] == scores[best] else top[1])
        score = float(scores[best])
        if score < self.threshold or score - second < self.margin:
            return None
        question, answer = self.records[best]
        return {"score": score, "question": question, "answer": answer}
