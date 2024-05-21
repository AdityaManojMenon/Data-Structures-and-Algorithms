from typing import TypeVar, List
import unittest

T = TypeVar('T')


class MinHeap:

    def __init__(self):
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def _swap(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def empty(self) -> bool:
        return len(self) == 0

    def top(self) -> T:
        if not self.empty():
            return self.data[0]

    def left_child_index(self, index: int) -> int:
        return 2 * index + 1 if 2 * index + 1 < len(self.data) else None

    def right_child_index(self, index: int) -> int:
        return 2 * index + 2 if 2 * index + 2 < len(self.data) else None

    def parent_index(self, index: int) -> int:
        return (index - 1) // 2 if index > 0 else None

    def min_child_index(self, index: int) -> int:
        left_i = self.left_child_index(index)
        right_i = self.right_child_index(index)
        if left_i is not None and (right_i is None or self.data[left_i] < self.data[right_i]):
            return left_i
        return right_i

    def percolate_up(self, index: int) -> None:
        parent_i = self.parent_index(index)
        if parent_i is not None and self.data[index] < self.data[parent_i]:
            self._swap(index, parent_i)
            self.percolate_up(parent_i)

    def percolate_down(self, index: int) -> None:
        min_child_i = self.min_child_index(index)
        if min_child_i is not None and self.data[index] > self.data[min_child_i]:
            self._swap(index, min_child_i)
            self.percolate_down(min_child_i)

    def add(self, key: T) -> None:
        self.data.append(key)
        self.percolate_up(len(self.data) - 1)

    def remove(self) -> T:
        if self.empty():
            return None
        min_elem = self.data[0]
        self.data[0] = self.data[-1]
        self.data.pop()
        if not self.empty():
            self.percolate_down(0)
        return min_elem

    def build_heap(self) -> None:
        for i in range(len(self.data) // 2 - 1, -1, -1):
            self.percolate_down(i)

    def is_min_heap(self, index=0) -> bool:
        if index >= len(self.data):
            return True

        left_i = self.left_child_index(index)
        right_i = self.right_child_index(index)
        
        left_valid = left_i is not None and left_i < len(self.data) and self.data[left_i] >= self.data[index]
        right_valid = right_i is not None and right_i < len(self.data) and self.data[right_i] >= self.data[index]

        return (left_valid or left_i is None) and (right_valid or right_i is None) and \
               (left_i is None or self.is_min_heap(left_i)) and \
               (right_i is None or self.is_min_heap(right_i))
