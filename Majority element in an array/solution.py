# SOLUTION 1
import collections
from typing import List


# First Problem
def find_majority_element_bf(nums: List[int]) -> int:
    """
    :param nums: int container to look up
    :return: frequent value in array
    :pre-cond: there will be at least one elem in list
    """
    nums.sort()
    max, ans = 0, 0
    N, i = len(nums), 1
    if N == 1:
        return nums[0]
    while i < N:
        count = 1
        while i < N and nums[i] == nums[i - 1]:
            count += 1
            i += 1
        if count > max:
            max = count
            ans = nums[i - 1]
        i += 1
    if max > (N // 2):
        return ans


def find_majority_element_v1(self, nums: List[int]) -> int:
    nums.sort()
    return nums[len(nums) // 2]


def find_majority_element_v2(self, nums: List[int]) -> int:
    sums = {}
    for n in nums:
        if n not in sums:
            sum[n] = 1
        else:
            sum[n] += 1
        if sums[n] > len(nums) // 2:
            return n


def find_majority_element_v3(self, nums: List[int]) -> int:
    count = collections.Counter(nums)
    return max(count.keys(), key=count.get)


def find_majority_element_Boyer_Moore(self, nums: List[int]) -> int:
    candidate = 0
    count = 0
    for element in nums:
        if count == 0:
            candidate = element
        if element == candidate:
            count += 1
        else:
            count -= 1
    return candidate


# Second Problem
class Logger(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._msg_dict = {}

    def should_print_message(self, timestamp, message) -> bool:
        """
        :return  true if the message should be printed in the given timestamp, otherwise returns false.
        """
        if message not in self._msg_dict:
            self._msg_dict[message] = timestamp
            return True

        if timestamp - self._msg_dict[message] >= 10:
            self._msg_dict[message] = timestamp
            return True 
        else:
            return False
