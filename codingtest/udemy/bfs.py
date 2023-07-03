from bst import BinarySearchTree, Node

class BFSBinarySearchTree(BinarySearchTree):
    def __init__(self) -> None:
        super().__init__()
    
    def BFS(self):
        curr = self.root 
        queue = [] # node append 
        results = [] # value append
        queue.append(curr)
        
        while queue:
            curr = queue.pop(0) # pop first element in the list
            results.append(curr.value)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return results

my_tree = BFSBinarySearchTree()
my_tree.insert(47)
my_tree.insert(21)
my_tree.insert(76)
my_tree.insert(18)
my_tree.insert(27)
my_tree.insert(52)
my_tree.insert(82)

print(my_tree.BFS())