# 오른쪽 엘리먼트랑 비교해서 내가 더 크면 쭉쭉 오른쪽꺼랑 swap해줌
# 점점 큰 엘리먼트를 오른쪽에 먼저 보낸다는 게 버블같아서인가?
def bubble_sort(list):
    for i in range(len(list) - 1, 0, -1):
        for j in range(i):
            if list[j] > list[j + 1]:
                temp = list[j]
                list[j] = list[j + 1]
                list[j + 1] = temp
    return list


my_list = [3, 1, 5, 2, 4]
print(bubble_sort(my_list))
