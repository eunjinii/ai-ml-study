def insertion_sort(list):
    for i in range(1, len(list)):
        temp = list[i]
        j = i - 1
        while list[j] > temp and j > -1:
            list[j + 1] = list[j]  # 서로 바꾼다기보단 옆으로 한 칸 간다는 의미에서.
            list[j] = temp
            j -= 1
    return list
# bubble sort랑 비슷한데, 버블은 for문 두개로 돌면서 swap한다면
# 이건 왼쪽이랑 비교해서 필요한 때만 오른쪽 index로 땡겨주는 개념인 것 같음.
# 따라서 베스트 케이스에는 O(N) time complexity가 소요됨
# i인덱스 값을 temp로 빼 놨다가, 오른쪽 인덱스 값들을 필요한 만큼 다 옮기고 나서 빈자리에 넣어준다는 의미에서 insertion sort 같음


my_list = [4, 5, 1, 13, 6, 9]
print(insertion_sort(my_list))
