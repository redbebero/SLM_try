import argparse
import json
from pathlib import Path


NAMES = ["민서", "지훈", "서연", "도윤", "하윤", "현우"]


def row(category, template_id, index, question, solution, answer):
    return {
        "id": f"owned-{category}-{template_id}-{index}",
        "category": category,
        "question": question,
        "solution": solution,
        "answer": str(answer),
        "template_id": template_id,
    }


def build():
    rows = []
    categories = ["arithmetic", "multi_step", "comparison", "document_qa", "state_change"]
    for i, name in enumerate(NAMES):
        for template in range(50):
            category = categories[template // 10]
            local_template = template % 10
            template_id = f"{category}-{local_template}"
            value = 3 + i + local_template
            if category == "arithmetic":
                answer = value * (i + 2)
                question = f"{name}는 한 상자에 {value}개씩 들어 있는 상자 {i + 2}개를 가지고 있습니다. 모두 몇 개인가요?"
                solution = f"상자 수와 개수를 곱합니다: {value} × {i + 2} = {answer}."
            elif category == "multi_step":
                answer = value + 4 * (i + 1) - 2
                question = f"{name}가 연필 {value}자루를 가지고 있었습니다. {4 * (i + 1)}자루를 받고 2자루를 썼다면 몇 자루가 남나요?"
                solution = f"{value} + {4 * (i + 1)} - 2 = {answer}입니다."
            elif category == "comparison":
                left, right = value + 7, value + 2
                answer = left - right
                question = f"A 상자에는 물건이 {left}개, B 상자에는 {right}개 있습니다. A 상자가 몇 개 더 많나요?"
                solution = f"차이를 계산합니다: {left} - {right} = {answer}."
            elif category == "document_qa":
                answer = 2010 + value
                question = f"다음 문서를 읽으세요. ‘해솔 도서관은 {answer}년에 문을 열었고 매주 월요일에 쉽니다.’ 도서관은 언제 문을 열었나요?"
                solution = f"문서에서 개관 연도 {answer}년을 찾습니다."
            else:
                answer = value + 12 - (i + 1)
                question = f"창고에 물품이 {value}개 있었습니다. {12}개가 들어오고 {i + 1}개가 나갔습니다. 현재 몇 개인가요?"
                solution = f"현재 수량은 {value} + 12 - {i + 1} = {answer}개입니다."
            rows.append(row(category, template_id, i, question, solution, answer))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
