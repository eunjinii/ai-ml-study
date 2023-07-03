def max_subarray(my_list):
    if not my_list:
        return 0

    sum_list = []
    sum_list.append(my_list[0])

    for i in range(1, len(my_list)):
        sum_list.append(max(sum_list[i - 1] + my_list[i], my_list[i]))

    result = max(sum_list)
    return result


my_list_1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]  # 6
print(max_subarray(my_list_1))

my_list_2 = [1, 2, 3, -4, 5, 6]  # 13
print(max_subarray(my_list_2))
