class Node:
  def __init__(self, value) -> None:
    self.value = value
    self.next = None 
    
class Stack:
  def __init__(self, value) -> None:
    new_node = Node(value)
    self.top = new_node 
    self.height = 1
    
  def print_stack(self):
    temp = self.top
    while (temp):
      print(temp.value)
      temp = temp.next
      
  def push(self, value):
    new_node = Node(value)
    if self.top is None:
      self.top = new_node
      return True
    new_node.next = self.top 
    self.top = new_node
    self.height += 1
    return True
  
  def pop(self):
    if self.top is None: 
      return None 
    
    # (1) -> (2) -> (3) -> (4)
    temp = self.top
    self.top = self.top.next 
    temp.next = None
    
    self.height -= 1
    return temp

    
    
    
my_stack = Stack(2)
my_stack.push(1)
my_stack.push(80)
my_stack.push(51)
print(my_stack.pop(), '\n')
my_stack.print_stack()