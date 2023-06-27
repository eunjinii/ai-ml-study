class Node:
    def __init__(self, value: float) -> None:
        self.value = value
        self.next = None
        self.prev = None


class DoublyLinkedList():
    def __init__(self, value) -> None:
        new_node = Node(value)
        self.head = new_node
        self.tail = new_node
        self.length = 1

    def print_list(self):
        temp = self.head
        while(temp):
            print(temp.value)
            temp = temp.next

    def append(self, value) -> bool:
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            self.length += 1
            return True
        self.tail.next = new_node
        new_node.prev = self.tail
        self.tail = new_node
        self.length += 1
        return True

    def pop(self) -> Node:
        if self.head is None:
            return None
        temp = self.tail
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            temp.prev = None
            self.tail.next = None
        self.length -= 1
        return temp

    def prepend(self, value) -> bool:
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node
        self.length += 1
        return True

    def pop_first(self) -> Node:
        if self.head is None:
            return None
        temp = self.head
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            temp.next = None
            self.head.prev = None
        self.length -= 1
        return temp

    def get(self, index) -> Node:
        if index < 0 or index >= self.length:
            return None

        if index < self.length / 2:  # index가 전체 길이의 절반보다 작을때만 앞에서 loop
            temp = self.head
            for _ in range(index):
                temp = temp.next
        else:
            temp = self.tail
            for _ in range(self.length - 1, index, -1):
                temp = temp.prev
        return temp

    def set_value(self, index, value) -> bool:
        target = self.get(index)
        if target:
            target.value = value
            return True
        return False

    def insert(self, index, value) -> bool:
        new_node = Node(value)
        if index < 0 or index > self.length:
            return False
        if index == 0:
            self.prepend(value)
        if index == self.length:
            self.append(value)

        after = self.get(index)
        before = after.prev

        new_node.next = after
        after.prev = new_node
        new_node.prev = before
        before.next = new_node

        self.length += 1
        return True

    def remove(self, index) -> Node:
        if index < 0 or index >= self.length:
            return None
        if index == 0:
            return self.pop_first()
        if index == self.length - 1:
            return self.pop()

        temp = self.get(index)
        temp.next.prev = temp.prev
        temp.prev.next = temp.next
        temp.next = None
        temp.prev = None

        self.length -= 1
        return temp.value


my_doubly_linked_list = DoublyLinkedList(12)
my_doubly_linked_list.append(6)
my_doubly_linked_list.append(100)
my_doubly_linked_list.append(39)
# my_doubly_linked_list.pop()
print(my_doubly_linked_list.remove(2))


# my_doubly_linked_list.print_list()
