# 2023-06-26

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self, value):
        new_node = Node(value)
        self.head = new_node
        self.tail = new_node
        self.length = 1

    def append(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1
        return True

    def prepend(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.length += 1
        return True

    def pop(self):
        if self.length == 0:
            return None

        pre = self.head
        temp = self.head

        while (temp.next):
            pre = temp
            temp = temp.next

        self.tail = pre
        self.tail.next = None
        self.length -= 1

        if self.length == 0:
            self.head = None
            self.tail = None
        return temp

    def pop_first(self):
        if self.length == 0:
            return None

        temp = self.head
        self.head = self.head.next
        temp.next = None  # 기존 남아있는 연결 끊어줌
        self.length -= 1

        if self.length == 0:
            # self.head = None # 이 엣지 케이스에서는 이미 self.head.next에서 None을 바라보고 있음
            self.tail = None

        return temp

    def get(self, index):
        if index < 0 or index >= self.length:  # index 유효한지 확인해야 한다.
            return None
        temp = self.head
        for _ in range(index):
            temp = temp.next
        return temp

    def set(self, index, value):
        temp = self.get(index)  # get method reuse
        if temp:  # get method reuse
            temp.value = value
            return True
        return False

    def insert(self, index, value):
        if index < 0 or index > self.length:
            return False
        if self.length == 0:
            return self.prepend(value)  # 재활용
        if index == self.length:
            return self.append(value)  # 재활용

        new_node = Node(value)
        pre = self.get(index - 1)
        new_node.next = pre.next
        pre.next = new_node
        self.length += 1
        return True

    def remove(self, index):
        if index < 0 or index >= self.length:
            return None  # remove에서는 성공하면 Node를 return할 것이므로, 엣지케이스에서는 None
        if index == 0:
            return self.pop_first()
        if index == self.length - 1:
            return self.pop()
        pre = self.get(index - 1)
        target = pre.next  # better complexity than self.get(index)
        pre.next = target.next
        target.next = None
        self.length -= 1
        return target

    def reverse(self):  # 매우 어려움 순서 주의 before, temp, after
        temp = self.head
        self.head = self.tail
        self.tail = temp

        after = temp.next
        before = None

        for _ in range(self.length):
            after = temp.next
            temp.next = before
            before = temp  # 갭 메꾸기
            temp = after  # 갭 메꾸기

    def print_list(self):
        temp = self.head
        while temp is not None:
            print(temp.value)
            temp = temp.next


my_linked_list = LinkedList(5)
my_linked_list.append(12)
my_linked_list.append(9)
my_linked_list.prepend(58)
my_linked_list.pop()
my_linked_list.pop_first()
my_linked_list.set(1, 13)
my_linked_list.insert(1, 90)
my_linked_list.reverse()

my_linked_list.print_list()
