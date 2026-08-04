import argparse
import json
from pathlib import Path


NAMES = ["민서", "지훈", "서연", "도윤", "하윤", "현우", "유진", "서준"]


def _row(category, split, index, question, solution, answer):
    return {
        "id": f"kr-reason-{split}-{category}-{index:03d}",
        "category": category,
        "question": question,
        "solution": solution,
        "answer": str(answer),
        "template_id": f"kr-reason-{split}-{category}-{index:03d}",
        "verifier": "text" if isinstance(answer, str) else "integer",
    }


def _make(category, split, index):
    offset = index + (0 if split == "test" else 1000)
    name = NAMES[offset % len(NAMES)]
    if category == "arithmetic":
        unit = ["권", "자루", "개", "봉지"][offset % 4]
        each, groups = 3 + offset % 17, 2 + offset % 8
        answer = each * groups
        return _row(category, split, index, f"{name}는 {unit}을 한 묶음에 {each}{unit}씩 담은 묶음 {groups}개를 가지고 있습니다. 모두 몇 {unit}인가요?", f"{each} × {groups} = {answer}입니다.", answer)
    if category == "multi_step":
        start, received, used = 8 + offset % 23, 3 + offset % 11, 1 + offset % 6
        answer = start + received - used
        return _row(category, split, index, f"{name}는 물건을 {start}개 가지고 있었습니다. {received}개를 더 받고 {used}개를 사용했습니다. 지금 몇 개인가요?", f"{start} + {received} - {used} = {answer}입니다.", answer)
    if category == "comparison":
        left, right = 20 + offset % 31, 4 + offset % 19
        answer = left - right
        return _row(category, split, index, f"상자 A에는 물건이 {left}개 있고 상자 B에는 {right}개 있습니다. A가 B보다 몇 개 더 많나요?", f"{left} - {right} = {answer}입니다.", answer)
    if category == "state_change":
        start, incoming, outgoing = 10 + offset % 25, 5 + offset % 13, 2 + offset % 7
        answer = start + incoming - outgoing
        return _row(category, split, index, f"창고에 물품 {start}개가 있습니다. 오전에 {incoming}개가 들어오고 오후에 {outgoing}개가 나갔습니다. 남은 물품은 몇 개인가요?", f"{start} + {incoming} - {outgoing} = {answer}입니다.", answer)
    if category == "temporal_logic":
        day = ["월요일", "화요일", "수요일", "목요일", "금요일"][offset % 5]
        closed = ["월요일", "수요일", "금요일"][(offset // 5) % 3]
        answer = "아니요" if day == closed else "예"
        return _row(category, split, index, f"도서관은 {closed}요일에 쉽니다. 이번 방문일이 {day}요일이라면 도서관에 들어갈 수 있나요?", f"휴관일과 방문일을 비교하면 {answer}입니다.", answer)
    year = 2000 + offset % 24
    fact = ["해솔 도서관", "푸른 박물관", "한빛 체육관"][offset % 3]
    return _row(category, split, index, f"다음 문서를 읽으세요. {fact}은 {year}년에 문을 열었고 매주 일요일에 쉽니다. {fact}은 언제 문을 열었나요?", f"문서에서 개관 연도 {year}년을 찾습니다.", f"{year}년")


def build_suite():
    categories = ["arithmetic", "multi_step", "comparison", "state_change", "temporal_logic", "reading_inference"]
    test_rows, ood_rows = [], []
    for category in categories:
        test_rows.extend(_make(category, "test", index) for index in range(150))
        ood_rows.extend(_make(category, "ood", index) for index in range(50))
    return test_rows, ood_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    test_rows, ood_rows = build_suite()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test", test_rows), ("ood", ood_rows)):
        (args.output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"test {len(test_rows)} ood {len(ood_rows)}")


if __name__ == "__main__":
    main()
