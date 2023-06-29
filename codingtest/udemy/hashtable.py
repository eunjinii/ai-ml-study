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
    if len(string) == 0: return None
    if len(string) == 1: return string[0]
    
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
    
print( first_non_repeating_char('leetcode') )
print( first_non_repeating_char('hello') )
print( first_non_repeating_char('aabbcc') )

