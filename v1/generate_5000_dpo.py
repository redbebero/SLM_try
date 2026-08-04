import json
import random
from faker import Faker

def get_name(fake):
    fantasy_names = ["앨리스", "해리", "론", "헤르미온느", "토니", "피터", "나타샤", "간달프", "프로도", "아라곤", "레골라스", "루피", "조로", "나미", "아인슈타인", "뉴턴", "셜록"]
    return random.choice(fantasy_names) if random.random() < 0.2 else fake.name()

def get_location(fake):
    fantasy_locs = ["마법의 숲", "우주정거장", "하늘섬", "비밀 기지", "용암 동굴", "얼음 궁전", "요정의 연못", "시간 여행선", "차원 교차로", "블랙홀 근처", "고대 유적", "사막 오아시스", "엘프의 마을", "드워프 광산"]
    return random.choice(fantasy_locs) if random.random() < 0.3 else fake.city() + " " + fake.street_name()

def get_company(fake):
    fantasy_comps = ["마법부", "은하 공화국", "어벤져스 본부", "제다이 기사단", "요정 길드", "마탑", "드래곤 레어", "연금술사 협회", "시간관리국", "우주 연방"]
    return random.choice(fantasy_comps) if random.random() < 0.25 else fake.company()

def get_job(fake):
    fantasy_jobs = ["대마법사", "연금술사", "요정", "우주비행사", "검사", "기사", "궁수", "탐험가", "시간여행자", "드래곤 라이더"]
    return random.choice(fantasy_jobs) if random.random() < 0.2 else fake.job()

def generate_5000_samples(output_file="dpo_5000_samples.jsonl"):
    fake = Faker('ko_KR')
    
    items_countable = ["사과", "쿠키", "장난감", "동전", "사탕", "젤리", "초콜릿", "책", "연필", "지우개", "구슬", 
                       "블록", "배터리", "나사", "기어", "부품", "박스", "볼펜", "의자", "책상", "모니터", "컵", 
                       "접시", "빵", "케이크", "티켓", "서류", "우표", "스티커", "화분", "노트", "지갑", 
                       "마법석", "마나 포션", "엘릭서", "금화", "드래곤의 비늘", "별빛 조각"] # 판타지 아이템 섞음
                       
    places_geo = ["주차장", "창고", "경기장", "무대", "온실", "훈련장", "비행기 활주로", "옥상", "수영장", 
                  "거실", "운동장", "농장", "텃밭", "공원", "광장", "전시장", "공장 부지", "물류센터", "캠핑장",
                  "마법진", "비밀 정원", "우주선 갑판", "고대 신전"] # 판타지 공간 섞음
                  
    machine_types = ["엔진", "발전기", "모터", "로봇", "드론", "서버", "터빈", "펌프", "컴퓨터", "에어컨", "프린터", "히터",
                     "워프 드라이브", "마나 코어", "방어막 생성기", "차원 도약기", "레이저 포탑"] # 판타지 기계 섞음

    samples = []
    
    for _ in range(1000):
        # ---------------------------------------------------------
        # 1. Arithmetic
        # ---------------------------------------------------------
        template_type = random.choice(["collect", "buy", "travel", "read"])
        a, b = random.randint(15, 999), random.randint(15, 999)
        target = a + b
        
        if template_type == "collect":
            n1, n2 = get_name(fake), get_name(fake)
            item = random.choice(items_countable)
            loc = get_location(fake)
            prompt = f"{loc}에 방문한 {n1}와(과) {n2}의 이야기입니다. {n1}은(는) {item}을(를) {a}개 모았고, {n2}은(는) {item}을(를) {b}개 구했습니다. 두 사람이 모은 {item}은(는) 모두 몇 개일까요?"
            desc_a, desc_b = f"{n1}의 {item}", f"{n2}의 {item}"
        elif template_type == "buy":
            company = get_company(fake)
            item = random.choice(items_countable)
            prompt = f"{company}에서는 어제 {item}을(를) {a}개 납품했고, 오늘은 {item}을(를) {b}개 납품했습니다. 이틀 동안 납품된 {item}은(는) 총 몇 개일까요?"
            desc_a, desc_b = f"어제 납품된 {item}", f"오늘 납품된 {item}"
        elif template_type == "travel":
            n1 = get_name(fake)
            prompt = f"{n1}은(는) 출장(또는 탐험)을 떠났습니다. 첫째 날에는 {a}km를 이동했고, 둘째 날에는 {b}km를 이동했습니다. {n1}이(가) 이동한 총 거리는 몇 km일까요?"
            desc_a, desc_b = "첫째 날 이동 거리", "둘째 날 이동 거리"
        else:
            n1 = get_name(fake)
            job = get_job(fake)
            prompt = f"{job}인 {n1}은(는) 문서를 검토 중입니다. 오전에 {a}페이지를 읽었고, 오후에 {b}페이지를 읽었습니다. {n1}이(가) 오늘 읽은 문서는 총 몇 페이지일까요?"
            desc_a, desc_b = "오전에 읽은 페이지", "오후에 읽은 페이지"

        chosen = f"주어진 조건을 정리하겠습니다.\n- {desc_a}: {a}\n- {desc_b}: {b}\n\n<think>\n1단계: 두 값을 더합니다.\n{a} + {b} = {target}\n\n2단계: 계산 결과를 확인합니다.\n결과는 {target}이 맞습니다. ✓\n</think>\n\n정답은 {target}입니다."
        wrong_ans = target + random.choice([-10, -1, 1, 10])
        rejected = f"주어진 조건을 정리하겠습니다.\n- {desc_a}: {a}\n- {desc_b}: {b}\n\n<think>\n1단계: 두 값을 더합니다.\n{a} + {b}를 계산할 때 암산 실수를 하여 {wrong_ans}이(가) 됩니다.\n\n2단계: 계산 결과를 확인합니다.\n결과가 {wrong_ans}라고 착각합니다. ✓\n</think>\n\n정답은 {wrong_ans}입니다."
        samples.append({"problem_id": f"arithmetic_{_}", "category": "arithmetic", "prompt": prompt, "chosen": chosen, "rejected": rejected, "target": target})

        # ---------------------------------------------------------
        # 2. Logic Puzzle
        # ---------------------------------------------------------
        template_type = random.choice(["age", "company", "money"])
        times = random.randint(2, 6)
        x = random.randint(5, 50) 
        years = random.randint(2, 15) 
        diff = (times - 1) * (x + years)
        target = x
        
        if template_type == "age":
            n1, n2 = get_name(fake), get_name(fake)
            prompt = f"{n1}와(과) {n2}의 나이 퍼즐입니다. {n1}은(는) {n2}보다 {diff}살 더 많습니다. {years}년 후에는 {n1}의 나이가 {n2}의 나이의 {times}배가 됩니다. 현재 {n2}의 나이는 몇 살일까요?"
            desc_x, desc_y = f"{n2}의 나이", f"{n1}의 나이"
            add_word = "년 후"
        elif template_type == "company":
            c1, c2 = get_company(fake), get_company(fake)
            prompt = f"{c1}와(과) {c2}의 설립 연수(또는 활동 기간) 비교입니다. {c1}은(는) {c2}보다 {diff}년 먼저 설립(시작)되었습니다. 두 조직이 만들어진 지 {years}년이 더 지나면, {c1}의 연수가 {c2}의 연수의 {times}배가 됩니다. 현재 {c2}의 연수는 몇 년일까요?"
            desc_x, desc_y = f"{c2}의 연수", f"{c1}의 연수"
            add_word = "년 후"
        else:
            n1, n2 = get_name(fake), get_name(fake)
            prompt = f"{n1}은(는) {n2}보다 자산(또는 주식/보물)을 {diff}개 더 많이 가지고 있습니다. 두 사람이 각각 보상으로 {years}개씩 더 받게 된다면, {n1}의 자산이 {n2}의 자산의 {times}배가 됩니다. 현재 {n2}이(가) 가진 자산은 몇 개일까요?"
            desc_x, desc_y = f"{n2}의 자산 수", f"{n1}의 자산 수"
            add_word = "개 추가 후"

        chosen = f"변수를 정의하겠습니다.\n- 현재 {desc_x}: x\n- 현재 {desc_y}: x + {diff}\n\n<think>\n1단계: {years}{add_word}의 상태를 식으로 세웁니다.\n비교 대상 1: x + {years}\n비교 대상 2: x + {diff} + {years} = x + {diff + years}\n\n2단계: {times}배 조건을 적용합니다.\nx + {diff + years} = {times}(x + {years})\n\n3단계: 방정식을 풉니다.\nx + {diff + years} = {times}x + {times * years}\n{diff + years} - {times * years} = {times}x - x\n{diff + years - times * years} = {times - 1}x\nx = {target}\n\n4단계: 검증합니다.\n현재 값 {target}, 차이 {diff} ✓\n</think>\n\n정답은 {target}입니다."
        wrong_ans = target + random.choice([-2, 2])
        rejected = f"변수를 정의하겠습니다.\n- 현재 {desc_x}: x\n- 현재 {desc_y}: x + {diff}\n\n<think>\n1단계: {years}{add_word}의 상태를 식으로 세웁니다.\n비교 대상 1: x + {years}\n비교 대상 2: x + {diff} + {years} = x + {diff + years}\n\n2단계: {times}배 조건을 적용합니다.\nx + {diff + years} = {times}(x + {years})\n\n3단계: 방정식을 풉니다.\nx + {diff + years} = {times}x + {times * years}\n{times - 1}x = {diff + years - times * years}\n나눗셈이나 이항 과정에서 단순 실수를 하여 x = {wrong_ans}라고 착각합니다.\n\n4단계: 검증합니다.\n결과값이 {wrong_ans}로 도출되었습니다. ✓\n</think>\n\n정답은 {wrong_ans}입니다."
        samples.append({"problem_id": f"logic_{_}", "category": "logic_puzzle", "prompt": prompt, "chosen": chosen, "rejected": rejected, "target": target})

        # ---------------------------------------------------------
        # 3. Ratio
        # ---------------------------------------------------------
        template_type = random.choice(["job", "vote", "mix"])
        r1 = random.randint(2, 15)
        r2 = random.randint(2, 15)
        if r1 == r2: r2 += 1
        multiplier = random.randint(10, 100)
        total = (r1 + r2) * multiplier
        target = max(r1, r2) * multiplier
        
        if template_type == "job":
            job1, job2 = get_job(fake), get_job(fake)
            company = get_company(fake)
            prompt = f"{company}에서 소속된 {job1}와(과) {job2}의 인원 비율은 {r1}:{r2}입니다. 전체 소속 인원이 {total}명일 때, 더 많은 인원을 차지하는 쪽의 사람 수는 몇 명일까요?"
        elif template_type == "vote":
            n1, n2 = get_name(fake), get_name(fake)
            loc = get_location(fake)
            prompt = f"{loc}의 대표 선거에서 {n1} 후보와 {n2} 후보의 득표(또는 지지) 비율이 {r1}:{r2}로 집계되었습니다. 유효 표 총합이 {total}표일 때, 더 많은 표를 얻은 후보의 득표수는 몇 표일까요?"
        else:
            item1, item2 = random.sample(items_countable, 2)
            prompt = f"어떤 상자(또는 공간) 안에 {item1}와(과) {item2}이(가) {r1}:{r2}의 비율로 섞여 있습니다. 두 물건의 총 개수가 {total}개일 때, 더 많이 들어있는 물건의 개수는 몇 개일까요?"

        chosen = f"주어진 조건을 정리하겠습니다.\n- 두 대상의 비율 = {r1}:{r2}\n- 총합 = {total}\n\n<think>\n1단계: 비례식을 세웁니다.\n비율의 합은 {r1} + {r2} = {r1+r2}입니다.\n\n2단계: 해당하는 값을 구합니다.\n더 큰 비율은 {max(r1, r2)}입니다.\n{total} × ({max(r1, r2)} / {r1+r2}) = {target}\n\n3단계: 검증합니다.\n정답이 {target}이 맞는지 확인했습니다. ✓\n</think>\n\n정답은 {target}입니다."
        wrong_ans = total // min(r1, r2)
        rejected = f"주어진 조건을 정리하겠습니다.\n- 두 대상의 비율 = {r1}:{r2}\n- 총합 = {total}\n\n<think>\n1단계: 비례식을 세웁니다.\n비율의 합은 {r1} + {r2} = {r1+r2}입니다.\n\n2단계: 값을 구합니다.\n비례배분을 할 때 총합인 {r1+r2}로 나누지 않고, 단일 비율로 나누는 실수를 범하거나 계산 실수를 하여 {wrong_ans}로 착각합니다.\n\n3단계: 검증합니다.\n계산된 값은 {wrong_ans}입니다. ✓\n</think>\n\n정답은 {wrong_ans}입니다."
        samples.append({"problem_id": f"ratio_{_}", "category": "ratio", "prompt": prompt, "chosen": chosen, "rejected": rejected, "target": target})

        # ---------------------------------------------------------
        # 4. Geometry
        # ---------------------------------------------------------
        geo = random.choice(places_geo)
        times_g = random.randint(2, 10)
        width = random.randint(10, 100)
        length = width * times_g
        area = width * length
        target = width
        
        prompt_templates = [
            f"{get_company(fake)}에서 직사각형 모양의 {geo}을(를) 설계(또는 건설) 중입니다. 가로 길이는 세로 너비의 {times_g}배입니다. 넓이가 {area}일 때, 세로 너비는 얼마일까요?",
            f"{get_name(fake)} 소유의 {geo} 도면이 있습니다. 도면상에서 한쪽 변의 길이가 다른 쪽 변의 {times_g}배라고 적혀 있습니다. 전체 면적이 {area}라면, 더 짧은 쪽 변의 길이는 얼마일까요?",
            f"{get_location(fake)}에 위치한 {geo}의 경계를 짓기 위해 측량을 했습니다. 길이가 너비보다 {times_g}배 길며, 총 넓이가 {area}인 직사각형 형태입니다. 너비는 얼마일까요?"
        ]
        prompt = random.choice(prompt_templates)
        
        chosen = f"변수를 정의하겠습니다.\n- 짧은 변(너비): w\n- 긴 변(길이): {times_g}w\n\n<think>\n1단계: 직사각형 넓이 공식을 적용합니다.\nw × {times_g}w = {area}\n{times_g}w² = {area}\n\n2단계: w²을 구합니다.\nw² = {area} ÷ {times_g} = {area // times_g}\n\n3단계: w를 구합니다.\nw = √{area // times_g} = {target}\n\n4단계: 검증합니다.\n너비 {target}, 길이 {target * times_g}\n{target} × {target * times_g} = {area} ✓\n</think>\n\n정답은 {target}입니다."
        wrong_ans = (area // times_g) // 2
        rejected = f"변수를 정의하겠습니다.\n- 짧은 변(너비): w\n- 긴 변(길이): {times_g}w\n\n<think>\n1단계: 직사각형 넓이 공식을 적용합니다.\nw × {times_g}w = {area}\n{times_g}w² = {area}\n\n2단계: w²을 구합니다.\nw² = {area} ÷ {times_g} = {area // times_g}\n\n3단계: w를 구합니다.\nw²의 값을 구한 뒤, 제곱근(루트)을 씌워야 하는데 실수로 2로 나누는 착각을 범합니다.\nw = {area // times_g} ÷ 2 = {wrong_ans}\n\n4단계: 검증합니다.\n계산 결과 너비는 {wrong_ans}로 도출됩니다. ✓\n</think>\n\n정답은 {wrong_ans}입니다."
        samples.append({"problem_id": f"geometry_{_}", "category": "geometry", "prompt": prompt, "chosen": chosen, "rejected": rejected, "target": target})

        # ---------------------------------------------------------
        # 5. Equation
        # ---------------------------------------------------------
        template_type = random.choice(["machine", "finance", "item"])
        x = random.randint(50, 300)
        y = random.randint(10, x-1)
        hap = x + y
        cha = 2 * x - y
        target = x
        
        if template_type == "machine":
            company = get_company(fake)
            mach = random.choice(machine_types)
            prompt = f"{company}에서 두 개의 {mach} A와 B를 가동합니다. A와 B의 출력을 더하면 {hap}입니다. A 출력의 2배에서 B 출력을 빼면 {cha}입니다. {mach} A의 출력은 얼마일까요?"
        elif template_type == "finance":
            n1 = get_name(fake)
            prompt = f"{n1}은(는) 두 종류의 자산(또는 마나/수익금)을 관리 중입니다. 자산 1과 자산 2를 합치면 {hap}입니다. 자산 1의 2배에서 자산 2를 빼면 {cha}입니다. 자산 1의 값은 얼마일까요?"
        else:
            item1, item2 = random.sample(items_countable, 2)
            loc = get_location(fake)
            prompt = f"{loc}의 상점에서 {item1}와(과) {item2}의 가치를 흥정하고 있습니다. 두 물건의 가치를 합치면 {hap}입니다. {item1} 가치의 2배에서 {item2} 가치를 빼면 {cha}입니다. {item1}의 가치는 얼마일까요?"

        chosen = f"주어진 조건을 정리하겠습니다.\n- 첫 번째 값: x, 두 번째 값: y\n- x + y = {hap}\n- 2x - y = {cha}\n\n<think>\n1단계: 두 식을 더합니다.\n(x + y) + (2x - y) = {hap} + {cha}\n3x = {hap + cha}\n\n2단계: x를 구합니다.\nx = {hap + cha} ÷ 3 = {target}\n\n3단계: 검증합니다.\n결과가 {target}이 맞습니다. ✓\n</think>\n\n정답은 {target}입니다."
        wrong_ans = target + random.choice([-5, -2, 2, 5])
        rejected = f"주어진 조건을 정리하겠습니다.\n- 첫 번째 값: x, 두 번째 값: y\n- x + y = {hap}\n- 2x - y = {cha}\n\n<think>\n1단계: 두 식을 더합니다.\n(x + y) + (2x - y) = {hap} + {cha}\n3x = {hap + cha}\n\n2단계: x를 구합니다.\n나눗셈을 암산하다가 실수하여 결과가 {wrong_ans}라고 착각합니다.\n\n3단계: 검증합니다.\n계산 결과는 {wrong_ans}입니다. ✓\n</think>\n\n정답은 {wrong_ans}입니다."
        samples.append({"problem_id": f"equation_{_}", "category": "equation", "prompt": prompt, "chosen": chosen, "rejected": rejected, "target": target})

    random.shuffle(samples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ Faker의 현실성(70~80%)과 판타지적 상상력(20~30%)이 완벽히 혼합된 5000개의 DPO 샘플이 {output_file}에 생성되었습니다!")

if __name__ == '__main__':
    generate_5000_samples()
