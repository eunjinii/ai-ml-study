from bst import BinarySearchTree

# 이미 탐색한 노드에서 어떻게 상위 노드로 가는지 이해 안 간다면 Stack 구조를 생각하면 됨.
# 한 함수 코드가 끝나고 리턴이 되면 stack에서 사라지고 상위 블록 차례가 된다.


class DFSBinarySearchTree(BinarySearchTree):
    def __init__(self) -> None:
        super().__init__()

    def dfs_pre_order(self):  # 탐색하며 닿는대로 write
        result = []

        def traverse(current_node):
            result.append(current_node.value)
            if current_node.left:
                traverse(current_node.left)
            if current_node.right:
                traverse(current_node.right)
        traverse(self.root)
        return result

    def dfs_post_order(self):  # 돌아오며 끝부터 write
        result = []

        def traverse(current_node):
            if current_node.left:
                traverse(current_node.left)
            if current_node.right:
                traverse(current_node.right)
            result.append(current_node.value)

        traverse(self.root)
        return result

    def dfs_in_order(self):  # 돌아오며 닿는대로 write. 작은 수부터 리스트에 들어감
        result = []

        def traverse(current_node):
            if current_node.left:
                traverse(current_node.left)
            result.append(current_node.value)  # 왼쪽만 다 탐색한 후에 write
            if current_node.right:
                traverse(current_node.right)

        traverse(self.root)
        return result


my_tree = DFSBinarySearchTree()
my_tree.insert(47)
my_tree.insert(21)
my_tree.insert(76)
my_tree.insert(18)
my_tree.insert(27)
my_tree.insert(52)
my_tree.insert(82)

print(my_tree.dfs_in_order())
