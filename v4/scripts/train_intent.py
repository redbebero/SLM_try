"""Train a tiny HRM jamo dialogue-intent specialist."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_intent import HRMIntentNet
from tokenizer import KoJamoTokenizer


RESPONSES = {
    "capital": "서울입니다.",
    "king": "세종대왕입니다.",
    "cat": "포유류인 동물입니다.",
    "python": "프로그래밍 언어입니다.",
    "moon": "달입니다.",
    "greet_help": "안녕하세요! 무엇을 도와드릴까요?",
    "greet_welcome": "저도 반가워요! 편하게 말씀해 주세요.",
    "greet_well": "네, 잘 지냈어요. 무엇을 도와드릴까요?",
    "advice_ask": "모르는 것은 편하게 물어보면 됩니다.",
    "advice_study": "작은 목표를 세우고 꾸준히 연습해 보세요.",
    "advice_mistake": "원인을 살펴보고 다시 시도해 보세요.",
    "advice_tasks": "중요한 일부터 차례대로 정리해 보세요.",
    "context_city": "__CITY__입니다.",
    "context_food": "__FOOD__입니다.",
    "unknown": "현재 확인할 수 있는 정보가 없습니다.",
}

UNKNOWN_EXAMPLES = (
    "오늘 서울의 날씨가 어때?", "지금 한국의 대통령은 누구야?",
    "화성에 생명체가 있나요?", "양자역학을 쉽게 설명해줘.",
    "김치찌개를 만드는 방법은?", "새로운 친구를 사귀는 방법을 알려줘.",
    "내일 어떤 일이 일어날까요?", "우주가 얼마나 큰지 알려줘.",
    "주식 투자는 어떻게 시작하나요?", "이 문장을 영어로 번역해줘.",
    "건강을 위해 어떤 운동을 해야 하나요?", "왜 하늘은 파란색인가요?",
    "좋은 책을 추천해줘.", "컴퓨터가 고장 났는데 어떻게 해요?",
    "이번 주말 여행지를 골라줘.",
)


def label_for(question, answer):
    q = question.replace(" ", "")
    if "서울" in answer:
        return "capital"
    if "세종" in answer:
        return "king"
    if "포유" in answer:
        return "cat"
    if "프로그래밍" in answer:
        return "python"
    if answer.startswith("달"):
        return "moon"
    if "도와" in answer:
        return "greet_help"
    if "반가" in answer:
        return "greet_welcome"
    if "잘 지냈" in answer:
        return "greet_well"
    if "물어" in answer:
        return "advice_ask"
    if "목표" in answer or "연습" in answer:
        return "advice_study"
    if "다시" in answer:
        return "advice_mistake"
    if "중요한 일" in answer:
        return "advice_tasks"
    if "사는" in q or "거주" in q or "사는곳" in q:
        return "context_city"
    if "산 것은" in q or "산음식" in q or "산과일" in q:
        return "context_food"
    return None


def load_pairs(path):
    pairs = []
    text = Path(path).read_text(encoding="utf-8")
    for block in re.split(r"\n\s*\n", text):
        q = re.search(r"(?m)^Q:\s*(.+)$", block)
        a = re.search(r"(?m)^A:\s*(.+)$", block)
        if not q or not a:
            continue
        label = label_for(q.group(1), a.group(1))
        if label:
            pairs.append((q.group(1), label))
    pairs.extend((question, "unknown") for question in UNKNOWN_EXAMPLES)
    return list(dict.fromkeys(pairs))


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, labels, max_length):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        for question, label in pairs:
            encoded = tokenizer.encode(question)[:max_length]
            self.items.append((encoded, labels.index(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    length = max(item[0].size(0) for item in batch)
    xs, masks, ys = [], [], []
    for x, y in batch:
        pad = length - x.size(0)
        xs.append(torch.cat([x, torch.zeros(pad, x.size(1), dtype=x.dtype)]))
        masks.append(torch.cat([torch.ones(x.size(0)), torch.zeros(pad)]))
        ys.append(y)
    return torch.stack(xs), torch.stack(masks), torch.tensor(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KoJamoTokenizer()
    labels = list(RESPONSES)
    pairs = load_pairs(args.data)
    dataset = IntentDataset(pairs, tokenizer, labels, 160)
    train_n = max(1, int(len(dataset) * 0.9))
    train, val = random_split(dataset, [train_n, len(dataset) - train_n])
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, collate_fn=collate)
    model = HRMIntentNet(tokenizer.get_vocab_sizes(), len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, mask, y in val_loader:
                pred = model(x.to(device), mask.to(device)).argmax(-1).cpu()
                correct += int((pred == y).sum())
                total += y.numel()
        accuracy = correct / max(1, total)
        print(f"Intent epoch {epoch} | val={accuracy:.4f} | samples={len(dataset)}")
        if accuracy >= best:
            best = accuracy
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save({"model": model.state_dict(), "labels": labels,
                        "responses": RESPONSES}, args.output)


if __name__ == "__main__":
    main()
