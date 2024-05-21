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
        INSERT DOCSTRINGS HERE!
        """
        pass

    def push(self, val: T, back: bool = True) -> None:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def pop(self, back: bool = True) -> None:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def list_to_dll(self, source: List[T]) -> None:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def dll_to_list(self) -> List[T]:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def _find_nodes(self, val: T, find_first: bool = False) -> List[Node]:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def find(self, val: T) -> Node:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def find_all(self, val: T) -> List[Node]:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def _remove_node(self, to_remove: Node) -> None:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def remove(self, val: T) -> bool:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def remove_all(self, val: T) -> int:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass

    def reverse(self) -> None:
        """
        INSERT DOCSTRINGS HERE!
        """
        pass


def dream_escaper(dll: DLL) -> DLL:
    """
    INSERT DOCSTRING HERE!
    """
    pass