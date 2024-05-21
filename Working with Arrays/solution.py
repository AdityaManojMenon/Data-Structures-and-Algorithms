from typing import TypeVar, Tuple, List  # For use in type hinting


class CodingC4:

    # Brute Force Solution
    def two_sum_array_quadratic(self, nums: List[int], target: int) -> List[int]:
        # Iterate over the `nums` list
        a = len(nums)
        for i in range(a):
        # Iterate over the remaining elements in the `nums` list, starting from the next index
          for j in range (i + 1, a):
        # If the current value and the next value add up to the target,
        # return the current index and the next index
            if nums[i] + nums[j] == target:
                return [i, j]
        return []

    def two_sum_array_v1(self, nums: List[int], target: int) -> List[int]:
        # Create a dictionary to store the value and its index in the `nums` list
        value_index_map = {}
        # Iterate over the `nums` list and store each value and its index in the dictionary
        for i, num in enumerate(nums):
          value_index_map[num] = i
        # Iterate over the `nums` list again and check
        # if the complement (target - current value) is in the dictionary
        for i, num in enumerate(nums):
          complement = target - num
        # If the complement is in the dictionary and its index is not the current index,
        # return the current index and the index of the complement
          if complement in value_index_map and value_index_map[complement] != i:
              return [i, value_index_map[complement]]
        return []

    def two_sum__array_v2(self, nums: List[int], target: int) -> List[int]:
        # Create a dictionary to store the complement and its index in the `nums` list
        complement_map = {}
        # Iterate over the `nums` list
        for i, num in enumerate(nums):
        # Calculate the complement (target - current value)
          complement = target - num
        # If the complement is in the dictionary, return the current index and the index of the complement
          if complement in complement_map:
              return [complement_map[complement], i]
        # If the complement is not in the dictionary, store the current value and its index in the dictionary
          complement_map[num] = i
        return []


    def two_sum_array_sorted(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
          current_sum = numbers[left] + numbers[right]
          if current_sum == target:
              return [left + 1, right + 1]
          elif current_sum < target:
              left += 1
          else:
              right -= 1
        return []
