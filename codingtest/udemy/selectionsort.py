# i 돌면서, i 다음 인덱스들을 j로 돌며 제일 작은 값의 인덱스를 업뎃해줌
# 인덱스 업뎃하고 나면 i인덱스의 값이랑 비교해서 작을 때 i 인덱스의 값이랑 바꿈
# min_index를 골라잡아서 바꾼다는 의미로 selection sort인가 싶음
def selection_sort(list):
    for i in range(len(list) - 1):
        min_index = i
        for j in range(i + 1, len(list)):
            if list[j] < list[min_index]:
                min_index = j
        if i != min_index:
            temp = list[min_index]
            list[min_index] = list[i]
            list[i] = temp
    return list


my_list = [5, 1, 8, 23, 6]
print(selection_sort(my_list))
