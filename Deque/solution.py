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
        Returns the number of elements currently in the deque.
        :return: The size of the deque.
        """

        return self.size

    def is_empty(self) -> bool:
        """
        Checks if the deque is empty.
        :return: True if the deque is empty, False otherwise.
        """

        if self.size == 0:
          return True

    def front_element(self) -> Optional[T]:
        """
        Retrieves the element at the front of the deque without removing it.
        :return: The element at the front of the deque, or None if the deque is empty.
        """

        if self.is_empty() or self.front is None:
          return None
        else:
          return self.queue[self.front]

    def back_element(self) -> Optional[T]:
        """
        Retrieves the element at the back of the deque without removing it.
        :return: The element at the back of the deque, or None if the deque is empty.
        """

        if self.is_empty() or self.back is None:
          return None
        else:
          return self.queue[self.back]

    def grow(self) -> None:
        """
        Doubles the capacity of the deque to accommodate more elements. Existing elements are rearranged to maintain order.
        """
        new_capacity = self.capacity * 2
        new_queue = [None] * new_capacity
        if not self.is_empty():
          for i in range(self.size):
            new_queue[i] = self.queue[(self.front + i)%self.capacity]
        
        self.queue = new_queue
        self.capacity = new_capacity
        self.front = 0
        self.back = self.size - 1
      

    def shrink(self) -> None:
        """
        Halves the capacity of the deque if the size is one-fourth of the current capacity or less, with a minimum capacity of 4. This method helps in saving space when the deque is sparsely populated.
        """
        
        new_capacity = max(self.capacity // 2, 4)

        new_queue = [None]*new_capacity
        if self.size <= new_capacity:
          new_queue = [None]*new_capacity
          for i in range(self.size):
            new_queue[i] = self.queue[(self.front + i) % self.capacity]
          self.queue = new_queue
          self.capacity = new_capacity
          self.front = 0
          self.back = self.size - 1

        

    def enqueue(self, value: T, front: bool = True) -> None:
        """
        Adds a new element to either the front or the back of the deque, depending on the 'front' parameter. If the deque reaches its capacity, it is automatically grown.
        :param value: The value to be added to the deque.
        :param front: A boolean indicating where to add the new element (True for front, False for back).
        """

        if self.size == 0:
          # Initialize front and back to 0 if the deque is empty
          self.front = 0
          self.back = 0
          self.queue[self.front] = value
        elif front:
            self.front = (self.front - 1 + self.capacity) % self.capacity
            self.queue[self.front] = value
        else:
            self.back = (self.back + 1) % self.capacity
            self.queue[self.back] = value
        self.size += 1
        if self.size == self.capacity:
            self.grow()

    def dequeue(self, front: bool = True) -> Optional[T]:
        """
        Removes and returns an element from either the front or the back of the deque, depending on the 'front' parameter. The deque is automatically shrunk if necessary.
        :param front: A boolean indicating from where to remove the element (True for front, False for back).
        :return: The removed element, or None if the deque was empty.
        """

        if self.is_empty():
          return None

        removed_value = None
        if front:
            removed_value = self.queue[self.front]
            self.front = (self.front + 1) % self.capacity
        else:
            removed_value = self.queue[self.back]
            self.back = (self.back - 1 + self.capacity) % self.capacity

        self.size -= 1

        if self.size <= self.capacity // 4 and self.capacity // 2 >= 4:
            self.shrink()
            self.front = 0
            self.back = self.size - 1

        if self.size == 0:
            self.front, self.back = None, None

        return removed_value
            



def maximize_profits(profits: List[int], k: int) -> int:
    """
    Calculates the maximum profit that can be achieved by working on certain days, given a constraint of working at least once every 'k' days.

    :param profits: A list of integers representing the profit that can be made on each day.
    :param k: An integer representing the constraint of needing to work at least once every 'k' days.
    :return: The maximum profit achievable under the given constraints.
    """
    n = len(profits)
    if n <= 2:
        return sum(profits)
    
    # Adjust k for the 1-based index loop below, ensuring we do not exceed the bounds.
    k = min(k, n - 1)

    # Dynamic programming array to store the maximum profit until day i.
    dp = [0] * n
    dp[0] = profits[0]
    dp[1] = profits[0] + profits[1]

    for i in range(2, n):
        # Calculate profit for working on the current day, considering the constraint of working at least once every k days.
        dp[i] = profits[i] + max(dp[i-1], max(dp[max(0, i-k):i-1]))

    # Ensure to add the profit of the last day.
    return dp[-1]