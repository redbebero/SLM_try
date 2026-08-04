import sys

sys.path.insert(0, "scripts")

from korean_arithmetic_nlu import solve_natural_arithmetic


def test_addition_with_counter_and_context():
    assert solve_natural_arithmetic("사과 3개에 2개를 더하면 몇 개야?") == "3+2=5개입니다."
    assert solve_natural_arithmetic("편의점에서 3000원짜리 음료와 1500원짜리 과자를 샀어. 얼마를 냈을까?") == "3000+1500=4500원입니다."


def test_subtraction_from_remaining_items():
    assert solve_natural_arithmetic("사과 10개 중에서 3개를 먹었어. 몇 개 남아?") == "10-3=7개입니다."


def test_multiplication_with_bundle_expression():
    assert solve_natural_arithmetic("연필 4자루씩 3묶음이면 모두 몇 자루야?") == "4×3=12자루입니다."


def test_division_with_people_expression():
    assert solve_natural_arithmetic("친구 4명에게 12000원을 똑같이 나눠주면 한 명당 얼마야?") == "12000÷4=3000원입니다."
    assert solve_natural_arithmetic("귤 6개를 두 사람에게 똑같이 나누면 한 사람당 몇 개야?") == "6÷2=3개입니다."


def test_korean_money_units():
    assert solve_natural_arithmetic("7천 원에서 2천5백 원을 빼면 얼마가 남아?") == "7000-2500=4500원입니다."
    assert solve_natural_arithmetic("지갑에 5만원이 있었는데 1만2천원을 꺼냈어. 얼마 남았지?") == "50000-12000=38000원입니다."


def test_explicit_multiplication_and_multistep_change():
    assert solve_natural_arithmetic("모니터는 21인치 x 12인치이고 총 픽셀을 계산해줘.") == "21×12=252입니다."
    assert solve_natural_arithmetic("상자에 12개가 있고 7개를 더 넣은 뒤 4개를 꺼냈어.") == "12+7-4=15개입니다."


def test_fraction_of_two_groups():
    assert solve_natural_arithmetic("검은 구슬 90개의 1/6과 흰 구슬 51개의 1/3을 꺼냈어. 모두 몇 개야?") == "90×1/6+51×1/3=32개입니다."


def test_does_not_guess_without_operation():
    assert solve_natural_arithmetic("사과 3개와 바나나 2개가 있어.") is None
