class Node:
  def __init__(self, value) -> None:
    self.value = value
    self.next = None 

class Queue:
  def __init__(self, value) -> None:
    new_node = Node(value)
    self.head = new_node 
    self.tail = new_node
    self.length = 1
  
  def print_queue(self):
    temp = self.head 
    while (temp):
      print(temp.value)
      temp = temp.next
  
  def enqueue(self, value): 
    new_node = Node(value )
    if self.head is None: 
      self.head = new_node
      self.tail = new_node
    else: 
      self.tail.next = new_node 
      self.tail = new_node
    self.length += 1 
    return True
  
  def dequeue(self):
    if self.head is None:
      return None 
    temp = self.head 
    if self.length == 1:
      self.head = None 
      self.tail = None 
    else: 
      #       head
      # (1) -> (2) -> (3)
      # temp
      self.head = self.head.next 
      temp.next = None 
    self.length -= 1
    return temp

my_queue = Queue(3)
my_queue.enqueue(4)
my_queue.enqueue(5)
my_queue.enqueue(6)
my_queue.dequeue()
my_queue.dequeue()
my_queue.print_queue()