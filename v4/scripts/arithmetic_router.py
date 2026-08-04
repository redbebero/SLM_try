"""Small deterministic arithmetic specialist used to diagnose routing needs."""

import re


def solve(question):
    numbers = [int(value) for value in re.findall(r"\d+", question)]
    if len(numbers) < 2:
        return None
    if "덧셈" in question:
        result = numbers[0] + numbers[1]
    elif "뺄셈" in question:
        result = numbers[0] - numbers[1]
    elif "곱셈" in question:
        result = numbers[-2] * numbers[-1]
    elif "나눗셈" in question:
        result = numbers[0] // numbers[1]
    else:
        return None
    unit = "원" if "원" in question else "자루" if "자루" in question else "개" if "개" in question else ""
    return f"{result}{unit}입니다."
