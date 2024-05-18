"""
Project 1
CSE 331 SS24 (Onsay)
Authors of DLL: Andrew McDonald, Alex Woodring, Andrew Haas, Matt Kight, Lukas Richters, 
                Anna De Biasi, Tanawan Premsri, Hank Murdock, & Sai Ramesh
solution.py
"""

from typing import TypeVar, List

# for more information on type hinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
Node = TypeVar("Node")  # represents a Node object (forward-declare to use in Node __init__)
DLL = TypeVar("DLL")


# pro tip: PyCharm auto-renders docstrings (the multiline strings under each function definition)
# in its "Documentation" view when written in the format we use here. Open the "Documentation"
# view to quickly see what a function does by placing your cursor on it and using CTRL + Q.
# https://www.jetbrains.com/help/pycharm/documentation-tool-window.html


class Node:
    """
    Implementation of a doubly linked list node.
    DO NOT MODIFY
    """
    __slots__ = ["value", "next", "prev", "child"]

    def __init__(self, value: T, next: Node = None, prev: Node = None, child: Node = None) -> None:
        """
        Construct a doubly linked list node.

        :param value: value held by the Node.
        :param next: reference to the next Node in the linked list.
        :param prev: reference to the previous Node in the linked list.
        :return: None.
        DO NOT MODIFY
        """
        self.next = next
        self.prev = prev
        self.value = value

        # The child attribute is only used for the application problem
        self.child = child

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        DO NOT MODIFY
        """
        return f"Node({str(self.value)})"

    __str__ = __repr__


class DLL:
    """
    Implementation of a doubly linked list without padding nodes.
    Modify only below indicated line.
    """
    __slots__ = ["head", "tail", "size"]

    def __init__(self) -> None:
        """
        Construct an empty doubly linked list.

        :return: None.
        DO NOT MODIFY
        """
        self.head = self.tail = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        DO NOT MODIFY
        """
        result = []
        node = self.head
        while node is not None:
            result.append(str(node))
            node = node.next
            if node is self.head:
                break
        return " <-> ".join(result)

    def __str__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        return repr(self)

    def __eq__(self, other: DLL) -> bool:
        """
        :param other: compares equality with this List
        :return: True if equal otherwise False
        DO NOT MODIFY
        """
        cur_node = self.head
        other_node = other.head
        while True:
            if cur_node != other_node:
                return False
            if cur_node is None and other_node is None:
                return True
            if cur_node is None or other_node is None:
                return False
            cur_node = cur_node.next
            other_node = other_node.next
            if cur_node is self.head and other_node is other.head:
                return True
            if cur_node is self.head or other_node is other.head:
                return False

    # MODIFY BELOW #
    # Refer to the classes provided to understand the problems better#

    def empty(self) -> bool:
        """
        Checks if the doubly linked list is empty.
        :return: True if the list is empty, else False.
        """
        if(self.size == 0):
            return True
        else:
            return False

    def push(self, val: T, back: bool = True) -> None:
        """
        Adds a new node with the specified value to the front or back of the list.
        :param val: The value to be added to the list.
        :param back: Determines whether the new node is added to the back (True) or front (False) of the list.
        """
        new_node = Node(val)
        
        if self.empty():
            self.head = self.tail = new_node
        elif back:
            # Append to the end of the list
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        else:
            # Prepend to the beginning of the list
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

        self.size += 1
    def pop(self, back: bool = True) -> None:
        """
        Removes a node from the front or back of the list.
        :param back: Determines whether the node is removed from the back (True) or front (False) of the list.
        """
        if(back == True):
            if (self.head == self.tail):
                self.head = self.tail = None
            else:
                self.tail = self.tail.prev
                self.tail.next = None
        else:
            if (self.head == self.tail):
                self.head = self.tail = None
            else:
                self.head = self.head.next
                self.head.prev = None
        self.size-=1

    def list_to_dll(self, source: List[T]) -> None:
        """
        Replaces the current list with a new list created from the provided source list.
        :param source: A standard Python list of elements to create the new doubly linked list.
        """
        self.head=self.tail=None
        self.size = 0
        for val in source:
            self.push(val)

    def dll_to_list(self) -> List[T]:
        """
        Converts the current doubly linked list into a standard Python list.
        :return: A list containing the values of the nodes in the doubly linked list.
        """
        
        res = []
        curr = self.head
        while curr:
            res.append(curr.value)
            curr = curr.next
        return res
        

    def _find_nodes(self, val: T, find_first: bool = False) -> List[Node]:
        """
        Finds nodes in the list with the specified value.
        :param val: The value to search for in the list.
        :param find_first: If True, returns a list containing only the first node found; if False, returns all matching nodes.
        :return: A list of nodes with the specified value.
        """
        
        res = []
        curr = self.head
        while curr:
            if (curr.value == val):
                res.append(curr)
                if(find_first == True):
                    break
            curr = curr.next
        return res
        

    def find(self, val: T) -> Node:
        """
        Finds the first node in the list with the specified value.
        :param val: The value to search for in the list.
        :return: The first node with the specified value, or None if no such node is found.
        """
        found_nodes = self._find_nodes(val, find_first=True)
        if found_nodes:
            return found_nodes[0]
        else:
            return None

    def find_all(self, val: T) -> List[Node]:
        """
        Finds all nodes in the list with the specified value.
        :param val: The value to search for in the list.
        :return: A list of all nodes with the specified value.
        """
        return self._find_nodes(val)

    def _remove_node(self, to_remove: Node) -> None:
        """
        Removes the specified node from the list.
        :param to_remove: The node to be removed.
        """
        if self.head is None:
          return None
        if not to_remove:
          return None
        #if the to_remove node is the only node in the list
        elif to_remove == self.head and to_remove == self.tail:
          self.head = self.tail = None
          self.size -=1
        #if the to_remove node is the head:
        elif to_remove == self.head:
          self.pop(False)
        #if the to_remove node is the tail:
        elif to_remove == self.tail:
          self.pop(True)
        #if the to_remove node is in the middle of the list
        else:
          to_remove.prev.next = to_remove.next
          to_remove.next.prev = to_remove.prev
          self.size -= 1


    def remove(self, val: T) -> bool:
        """
        Removes the first node with the specified value from the list.
        :param val: The value of the node to be removed.
        :return: True if a node was removed, False otherwise.
        """
        to_remove = self.find(val)
        if to_remove:
          self._remove_node(to_remove)
          return True
        else:
          return False

    def remove_all(self, val: T) -> int:
        """
        Removes all nodes with the specified value from the list.
        :param val: The value of the nodes to be removed.
        :return: The number of nodes removed.
        """
        count = 0 #initialized to zero to keep count
        curr = self.head
        while curr:
          next_node = curr.next  
          if curr.value == val:
            self._remove_node(curr)
            count += 1
          curr = next_node
        return count

    def reverse(self) -> None:
        """
        Reverses the doubly linked list in place.
        Swaps the 'next' and 'prev' references of all nodes and updates the 'head' and 'tail' of the list.
        """
        curr = self.head
        #swapping the nodes in the middle first
        while curr is not None:
          curr.next, curr.prev = curr.prev, curr.next
          curr = curr.prev
        #The swapping the head and tail
        self.head, self.tail = self.tail, self.head


def dream_escaper(dll: DLL) -> DLL:
    """
    Transforms a multilevel doubly linked list into a single-level list.
    Each node in the list may have a child, which is the head of another sublist. This function flattens 
    such multilevel lists by integrating the child sublists into the main list, maintaining the order.
    :param dll: A doubly linked list where each node may have a child pointing to another sublist.
    :return: A single-level doubly linked list with all sublists integrated.
    """

    flattened_list = DLL()
    nodes_to_visit = []
    current_node = dll.head

    while current_node or nodes_to_visit:
      if current_node:
        flattened_list.push(current_node.value)
        if current_node.child:
            if current_node.next:
                nodes_to_visit.append(current_node.next)
            current_node = current_node.child
        else:
            current_node = current_node.next
      elif nodes_to_visit:
        current_node = nodes_to_visit.pop()

    return flattened_list

   