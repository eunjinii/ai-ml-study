class HashTable:
    def __init__(self, size=7):
        self.data_map = [None] * size

    def __hash(self, key):
        my_hash = 0
        for letter in (key):
            my_hash = (my_hash + ord(letter) * 23) % len(self.data_map)
        return my_hash

    def print_table(self):
        for i, value in enumerate(self.data_map):
            print(i, ': ', value)

    def set_item(self, key, value):
        index = self.__hash(key)
        if self.data_map[index] == None:
            self.data_map[index] = []
        self.data_map[index].append([key, value])

    def get_item(self, key):
        index = self.__hash(key)
        if self.data_map[index] != None:
            for item in self.data_map[index]:
                if item[0] == key:
                    return item[1]
        return None

    def keys(self):
        all_keys = []
        for i in range(len(self.data_map)):
            if self.data_map[i] != None:
                for j in range(len(self.data_map[i])):
                    all_keys.append(self.data_map[i][j][0])
        return all_keys


my_hash_table = HashTable(7)
my_hash_table.set_item('bolts', 1400)
my_hash_table.set_item('washers', 50)
print(my_hash_table.keys())
my_hash_table.print_table()


def list_in_common(list1, list2):
    my_dict = {}
    for i in list1:
        my_dict[i] = True
    for j in list2:
        if j in my_dict:
            return True
    return False


list1 = [2, 3, 5]
list2 = [1, 6]

print(list_in_common(list1, list2))


def first_non_repeating_char(string):
    if len(string) == 0:
        return None
    if len(string) == 1:
        return string[0]

    repeat_check = {}
    result = []

    for i in string:
        # aaabbc
        # abccc
        if i not in repeat_check:
            repeat_check[i] = 1
            result.append(i)
        else:
            repeat_check[i] += 1
    for j in result:
        if repeat_check[j] == 1:
            return j


print(first_non_repeating_char('leetcode'))
print(first_non_repeating_char('hello'))
print(first_non_repeating_char('aabbcc'))


def group_anagrams(strings):
    ana_dict = {}
    for string in strings:
        canonical = ''.join(sorted(string))
        if canonical not in ana_dict:
            ana_dict[canonical] = []
        ana_dict[canonical].append(string)

    return list(ana_dict.values())


print("1st set:")
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))

print("\n2nd set:")
print(group_anagrams(["abc", "cba", "bac", "foo", "bar"]))

print("\n3rd set:")
print(group_anagrams(
    ["listen", "silent", "triangle", "integral", "garden", "ranged"]))


def two_sum(nums, target):
    # if len(nums) == 0:
    #     return nums # 불필요
    my_dict = {}
    for i, num in enumerate(nums):
        subtract = target - num
        if subtract in my_dict:
            return [my_dict[subtract], i]
        else:
            my_dict[num] = i

    return []


print(two_sum([2, 7, 11, 15], 9))
print(two_sum([3, 2, 4], 6))
print(two_sum([3, 3], 6))
print(two_sum([1, 2, 3, 4, 5], 10))
print(two_sum([1, 2, 3, 4, 5], 7))
print(two_sum([1, 2, 3, 4, 5], 3))
print(two_sum([], 0))


def has_unique_chars(string):
    # if len(string) == 0:
    #     return True

    duplicated = set()
    for letter in string:
        if (string.count(letter) > 1):
            return False
        else:
            duplicated.add(letter)
    return True


print(has_unique_chars('abcdefg'))  # should return True
print(has_unique_chars('hello'))  # should return False
print(has_unique_chars(''))  # should return True
print(has_unique_chars('0123456789'))  # should return True
print(has_unique_chars('abacadaeaf'))  # should return False


def find_pairs(arr1, arr2, target):
    pairs = []
    compare_set = set(arr1)
    for i in arr2:
        subtract = target - i
        if subtract in compare_set:
            pairs.append((subtract, i))
    return pairs


arr1 = [1, 2, 3, 4, 5]
arr2 = [2, 4, 6, 8, 10]
target = 7

pairs = find_pairs(arr1, arr2, target)
print(pairs)


def longest_consecutive_sequence(nums):
    # 연속된 num이 아니라 조합해서 연속적이면 됨
    num_set = set(nums)
    longest_seq_size = 0

    for i in nums:
        if i - 1 not in num_set:
            curr_num = i
            curr_seq_size = 1
            while curr_num + 1 in num_set:
                curr_num += 1
                curr_seq_size += 1

            longest_seq_size = max(curr_seq_size, longest_seq_size)

    return longest_seq_size


print(longest_consecutive_sequence([100, 4, 200, 1, 3, 2]))
