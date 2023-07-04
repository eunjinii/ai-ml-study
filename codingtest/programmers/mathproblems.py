# 약수 구하기
def get_cds_num(number, limit, power):
    divider = []
    for i in range(1, int(number ** (1/2)) + 1):  
        if number % i == 0:
            divider.append(i)
            # count += 1
            if i ** 2 != number: # 제곱근이 아니라면
                divider.append(number // i)
                # count += 1
    return divider

# 순열, 조합
from itertools import combinations

def get_combinations(number):
    answer = 0 # 두개의 합들을 dict로 만든 다음에, for문돌려서 차이를 찾음
    combi_list = combinations(number, 3)
    for combi in combi_list:
        if sum(combi) == 0:
            answer += 1
    return answer
# p = permutations(chars, 2)  # 순열 [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
# c = combinations(chars, 2)  # 조합 [('A', 'B'), ('A', 'C'), ('B', 'C')]