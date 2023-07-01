def quicksort(my_list):
    if len(my_list) <= 1:
        return my_list
    pivot = 0
    swap = 0
    for i in range(1, len(my_list)):
        if my_list[i] < my_list[pivot]:
            swap += 1
            my_list[swap], my_list[i] = my_list[i], my_list[swap]
    my_list[pivot], my_list[swap] = my_list[swap], my_list[pivot]
    return quicksort(my_list[:swap]) + [my_list[swap]] + quicksort(my_list[swap + 1:])

my_list = [4,6,1,7,3,2,5]
print(quicksort(my_list))

# 강의방식
def swap(my_list, index1, index2):
    temp = my_list[index1]
    my_list[index1] = my_list[index2]
    my_list[index2] = temp 

def pivot(my_list, pivot_index, end_index):
    swap_index = pivot_index
    
    for i in range(pivot_index + 1, end_index + 1):
        if my_list[pivot_index] < my_list[i]:
            swap_index += 1
            swap(my_list, swap_index, i)
    swap(my_list, pivot_index, swap_index)
    return swap_index

def quick_sort_helper(my_list, left_index, right_index):
    if left_index < right_index:
        pivot_index = pivot(my_list, left_index, right_index)
        quick_sort_helper(my_list, left_index, pivot_index - 1)
        quick_sort_helper(my_list, pivot_index + 1, right_index)
    return my_list

def quick_sort(my_list):
    return quick_sort_helper(my_list, 0, len(my_list))