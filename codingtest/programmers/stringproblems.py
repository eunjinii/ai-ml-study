# String substring 일부 잘라내기, replace + strip
def replace_problem(babbling):
    answer = 0
    for word in babbling:
        for j in ["aya", "ye", "woo", "ma"]:
            if j*2 not in word:
                word = word.replace(j, " ")
        if len(word.strip()) == 0:
            answer += 1
    return answer

babbling1 = ["aya", "yee", "u", "maa"]	
result1 = 1
babbling2  = ["ayaye", "uuu", "yeye", "yemawoo", "ayaayaa"]
result2 = 2
print(replace_problem(babbling2))