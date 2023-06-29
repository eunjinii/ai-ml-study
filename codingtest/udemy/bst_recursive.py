from bst import BinarySearchTree, Node


class RecursiveBST(BinarySearchTree):
    def __init__(self) -> None:
        super().__init__()

    def __r_contains(self, current_node, value):
        if current_node == None:
            return False
        if value == current_node.value:
            return True
        if value < current_node.value:
            return self.__r_contains(current_node.left, value)
        if value > current_node.value:
            return self.__r_contains(current_node.right, value)

    def r_contains(self, value):
        return self.__r_contains(self.root, value)

    def __r_insert(self, current_node, value):
        if current_node == None:
            return Node(value)
        if value < current_node.value:
            # 현재노드보다 작을 시, 현재노드.left에서 새로 확인해라
            # temp left로 움직이는 게 아니라 left에 바로 recursive 집어넣음
            current_node.left = self.__r_insert(current_node.left, value)
        if value > current_node.value:
            # 현재노드보다 클 시, 현재노드.right에서 새로 확인해라
            current_node.right = self.__r_insert(current_node.right, value)
        return current_node

    def r_insert(self, value):
        if self.root is None:
            self.root = Node(value)
        self.__r_insert(self.root, value)

    def __delete_node(self, current_node, value):  # insert와 비슷하다.
        if current_node == None:
            return None
        if value < current_node.value:
            current_node.left = self.__delete_node(current_node.left, value)
        elif value > current_node.value:
            current_node.right = self.__delete_node(current_node.right, value)
        else:  # 우리가 찾는 그 값일 때
            if current_node.left == None and current_node.right == None:
                return None  # leaf node이면 아무것도 안한다는 뜻. leaf 지워봤자 다른 노드는 영향 없기 때문
            elif current_node.left == None:
                current_node = current_node.right  # 그자리에 right 갖다붙임 = 삭제
            elif current_node.right == None:
                current_node = current_node.left  # 그자리에 left 갖다붙임 = 삭제
            else:  # 그자리에 right 중에서 최소값 복붙해서 갖다 놓고, 그 최소값 노드를 삭제함.
                sub_tree_min = self.min_value(current_node.right)
                current_node.value = sub_tree_min
                current_node.right = self.__delete_node(
                    current_node.right, sub_tree_min)  # right중에서 최소값 노드 삭제 위해 traverse
        return current_node

    def min_value(self, current_node):
        while current_node.left is not None:  # 왼쪽으로 temp 옮기듯이 끝까지 옮겨
            current_node = current_node.left
        return current_node.value


my_tree = RecursiveBST()
my_tree.r_insert(2)
my_tree.r_insert(61)
my_tree.r_insert(13)
my_tree.r_insert(45)
my_tree.r_insert(12)
my_tree.r_insert(80)
my_tree.r_insert(35)

print(my_tree.min_value(my_tree.root.right))
