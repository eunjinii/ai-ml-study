def merge(list1, list2):
    combined = []
    i = 0
    j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            combined.append(list1[i])
            i += 1
        else:
            combined.append(list2[j])
            j += 1
    while i < len(list1):
        combined.append(list1[i])
        i += 1
    while j < len(list2):
        combined.append(list2[j])
        j += 1

    return combined


# 이미 sort된 두 리스트를 merge할 때 씀
list1 = [2, 5, 7, 8, 41]
list2 = [4, 9, 10, 12, 87]

print(merge(list1, list2))


def merge_sort(my_list):
    if len(my_list) == 1:
        return my_list
    mid_index = int(len(my_list) / 2)  # for odd case

    # 분리 작업이 여기서 이루어지므로, merge_sort를 여기다 재귀적으로 작성.
    left = merge_sort(my_list[:mid_index])
    right = merge_sort(my_list[mid_index:])

    return merge(left, right)
