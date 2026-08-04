"""Conservative Korean-to-arithmetic normalizer for the hybrid runtime."""

import re


_NUMBER = (r"(?:[0-9][0-9,]*(?:만|천|백|십)?[0-9]*(?:천|백|십)?|"
           r"한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)")
_UNIT_WORDS = ("원", "개", "명", "사람", "자루", "장", "권", "마리", "번")
_KOREAN_NUMBERS = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}


def _number_value(token):
    token = token.replace(",", "")
    if token in _KOREAN_NUMBERS:
        return _KOREAN_NUMBERS[token]
    if token.isdigit():
        return int(token)
    total = 0
    rest = token
    for unit, multiplier in (("만", 10000), ("천", 1000), ("백", 100), ("십", 10)):
        if unit in rest:
            left, rest = rest.split(unit, 1)
            total += (int(left) if left else 1) * multiplier
    if rest:
        if not rest.isdigit():
            return None
        total += int(rest)
    return total or None


def _numbers(text):
    values = []
    for match in re.finditer(_NUMBER, text):
        # ``한 묶음/상자에 ...`` is an article-like quantity, not the first
        # operand in a later ``N개씩 M묶음`` multiplication.
        if match.group() == "한" and text[match.end():].startswith(("묶음", "상자", "꾸러미", "명", "명당", "사람당")):
            continue
        # The ``두`` in ``모두`` is not the number two.
        if match.group() == "두" and match.start() > 0 and text[match.start() - 1] == "모":
            continue
        value = _number_value(match.group())
        if value is not None:
            values.append((value, match.start(), match.end()))
    return values


def _unit(text, start, end):
    tail = text[end:end + 5]
    for unit in _UNIT_WORDS:
        if tail.startswith(unit):
            return unit
    return ""


def _render(expression, result, unit=""):
    return f"{expression}={result}{unit}입니다."


def _fraction_terms(text):
    pattern = rf"({_NUMBER})[가-힣]{{0,3}}의(\d+)/(\d+)"
    terms = []
    for match in re.finditer(pattern, text):
        base = _number_value(match.group(1))
        numerator, denominator = int(match.group(2)), int(match.group(3))
        if base is None or denominator == 0:
            return None
        terms.append((base, numerator, denominator))
    return terms


def solve_natural_arithmetic(question):
    """Return a verified answer, or ``None`` when the wording is ambiguous."""
    text = re.sub(r"\s+", "", question)

    fractions = _fraction_terms(text)
    if len(fractions) >= 2 and any(word in text for word in ("모두", "총", "합")):
        first, second = fractions[:2]
        left = first[0] * first[1] / first[2]
        right = second[0] * second[1] / second[2]
        result = int(left + right) if (left + right).is_integer() else left + right
        unit = next((u for u in _UNIT_WORDS if u in text), "")
        return _render(f"{first[0]}×{first[1]}/{first[2]}+{second[0]}×{second[1]}/{second[2]}", result, unit)

    found = _numbers(text)
    if len(found) < 2:
        return None
    first, second = found[:2]
    a, b = first[0], second[0]
    first_unit = _unit(text, first[1], first[2])
    second_unit = _unit(text, second[1], second[2])

    # Explicit multiplication has priority over the word ``총``.
    if len(found) == 2 and re.search(r"x|×|\*|곱|씩|묶음|배", text):
        return _render(f"{a}×{b}", a * b, first_unit or second_unit)

    # A three-number state transition: initial + added - removed.
    if (len(found) >= 3
            and re.search(r"더하|더해|더하고|더넣|넣고|추가|합치", text)
            and re.search(r"빼|꺼내|꺼냈|먹|남", text)):
        c = found[2][0]
        unit = first_unit or second_unit or _unit(text, found[2][1], found[2][2])
        return _render(f"{a}+{b}-{c}", a + b - c, unit)

    # Do not silently discard a third quantity from a word problem.
    if len(found) != 2:
        return None

    # Division. Identify the dividend by its money/item unit, not position.
    if re.search(r"나누|나눠|똑같이", text) or re.search(r"로", text) and "나누" in text:
        if first_unit in {"명", "사람"} and second_unit not in {"명", "사람"}:
            divisor, dividend, unit = a, b, second_unit
        else:
            divisor, dividend, unit = b, a, first_unit or second_unit
        if divisor == 0:
            return None
        return _render(f"{dividend}÷{divisor}", dividend // divisor, unit)

    # Subtraction cues must describe removal, not an unrelated place name.
    numeric_from = re.search(rf"{_NUMBER}[가-힣]{{0,3}}에서", text)
    removal = (numeric_from
               or re.search(r"중에서|빼|먹었|사용했|썼|쓰면|꺼내|꺼냈|줬|주면|남아|남은|남았", text))
    same_unit = bool(first_unit and second_unit and first_unit == second_unit)
    if removal and (same_unit or numeric_from or re.search(r"빼면|빼고", text)):
        return _render(f"{a}-{b}", a - b, first_unit or second_unit)

    # Addition requires an explicit addition cue or two same-unit amounts.
    same_currency_total = (len(found) == 2 and first_unit == second_unit == "원"
                           and any(word in text for word in ("총", "얼마")))
    explicit_addition = re.search(r"더하|더해|합치|합하면|모두", text)
    if explicit_addition or same_currency_total:
        return _render(f"{a}+{b}", a + b, first_unit or second_unit)

    return None
