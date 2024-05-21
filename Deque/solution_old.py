"""
Project 4: Circular Double-Ended Queue
CSE 331 SS24
David Rackerby
solution.py
"""

from typing import TypeVar, List, Optional

T = TypeVar('T')


class CircularDeque:
    """
    Representation of a Circular Deque using an underlying python list
    """

    __slots__ = ['capacity', 'size', 'queue', 'front', 'back']

    def __init__(self, data: Optional[List[T]] = None, front: int = 0, capacity: int = 4):
        """
        Initializes an instance of a CircularDeque
        :param data: starting data to add to the deque, for testing purposes
        :param front: where to begin the insertions, for testing purposes
        :param capacity: number of slots in the Deque
        """
        if data is None and front != 0:
            # front will get set to 0 by front_enqueue if the initial data is empty
            data = ['Start']
        elif data is None:
            data = []

        self.capacity: int = capacity
        self.size: int = len(data)
        self.queue: List[T] = [None] * capacity
        self.back: int = (self.size + front - 1) % self.capacity if data else None
        self.front: int = front if data else None

        for index, value in enumerate(data):
            self.queue[(index + front) % capacity] = value

    def __str__(self) -> str:
        """
        Provides a string representation of a CircularDeque
        'F' indicates front value
        'B' indicates back value
        :return: the instance as a string
        """
        if self.size == 0:
            return "CircularDeque <empty>"

        str_list = ["CircularDeque <"]
        for i in range(self.capacity):
            str_list.append(f"{self.queue[i]}")
            if i == self.front:
                str_list.append('(F)')
            elif i == self.back:
                str_list.append('(B)')
            if i < self.capacity - 1:
                str_list.append(',')

        str_list.append(">")
        return "".join(str_list)

    __repr__ = __str__

    # ============ Modify Functions Below ============#

    def __len__(self) -> int:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def is_empty(self) -> bool:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def front_element(self) -> Optional[T]:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def back_element(self) -> Optional[T]:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def grow(self) -> None:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def shrink(self) -> None:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def enqueue(self, value: T, front: bool = True) -> None:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass

    def dequeue(self, front: bool = True) -> Optional[T]:
        """
        PUT YOUR OWN DOCSTRING HERE
        """
        pass


def maximize_profits(profits: List[int], k: int) -> int:
    """
    PUT YOUR OWN DOCSTRING HERE
    """
    pass