from _testcapi import INT_MIN
from collections import deque


# PROBLEM 1 --> STACKS
# APPROACH 1
class MinStack1:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        # If the stack is empty, then the min valuez
        # must just be the first value we add
        if not self.stack:
            self.stack.append((x,x))
            return

        cur_min = self.stack[-1][1]
        self.stack.append((x, min(x, cur_min)))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]


# APPROACH II
class MinStack2:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# APPROACH III
class MinStack3:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x: int) -> None:
        # We always put the number onto the main stack.
        self.stack.append(x)
        # If the min stack is empty, or this number is smaller than
        # the top of the min stack, put it on with a count of 1.
        if not self.min_stack or x < self.min_stack[-1][0]:
            self.min_stack.append([x, 1])
        # Else if this number is equal to what's currently at the top
        # of the min stack, then increment the count at the top by 1.
        elif x == self.min_stack[-1][0]:
            self.min_stack[-1][1] += 1
        

    def pop(self) -> None:
        # If the top of min stack is the same as the top of stack
        # then we need to decrement the count at the top by 1.
        if self.stack.pop() == self.min_stack[-1][0]:
            self.min_stack[-1][1] -= 1
        # If the count at the top of min stack is now 0, then remove
        # that value as we're done with it.
            if self.min_stack[-1][1] == 0:
        # And like before, pop the top of the main stack.
                self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1][0]


# PROBLEM 2 SLIDING WINDOW
class WindowsQueues:
    def find_max_sum_brute(self, mylist, n, k):
        # Initialize result
        max_sum = float('-inf')
        # Consider all blocks
        # starting with i.
        for i in range(n - k + 1):
            cur_sum = sum(mylist[i:i+k])
            max_sum = max(max_sum, cur_sum)
        return max_sum

    def find_max_sum(self, data, n, k):
        # k must be smaller than n
        if n < k:
            return -1
        # Compute sum of first
        # window of size k
        window_sum = sum(data[:k])
        max_sum = window_sum

        for i in range(n - k):
            window_sum = window_sum - data[i] + data[i + k]
            max_sum = max(window_sum, max_sum)
        return max_sum


class MovingAverage1:

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._data = []

    def next(self, val: int) -> float:
        

        self._data.append(val)
        # calculate the sum of the moving window
        return sum(self._data[-self.window_size:]) / min(len(self._data), self.window_size)


class MovingAverage2:
    # About dequeue:
    # https://docs.python.org/2.5/lib/deque-objects.html
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        # number of elements seen so far
        self.window_sum = 0
        self.count = 0

    def next(self, val: int) -> float:
        self.count += 1
        # calculate the new sum by shifting the window
        tail = self.queue.popleft() if self.count > self.size else 0
        self.window_sum = self.window_sum - tail + val
        self.queue.append(val)
        # calculate the new sum by shifting the window
        # if the count exceeds the size, popleft ( means remove from front)
        return self.window_sum / min(self.size, self.count)