"""
Project 2 - Hybrid Sorting
CSE 331 Spring 2024
Aman T., Daniel B., David R., Matt W.
"""

from typing import TypeVar, List, Callable

T = TypeVar("T")  # represents generic type


# This is an optional helper function but HIGHLY recommended,  especially for the application problem!
def do_comparison(first: T, second: T, comparator: Callable[[T, T], bool], descending: bool) -> bool:
    """
    Compares two elements based on a comparator and order.
    :param first: The first element to compare.
    :param second: The second element to compare.
    :param comparator: Function to determine the order of elements.
    :param descending: True if order is descending, False otherwise.
    :return: Result of comparison based on the specified order.
    """
    if descending == True:
        return comparator(second,first)
    else:
        return comparator(first,second)


def selection_sort(data: List[T], *, comparator: Callable[[T, T], bool] = lambda x, y: x < y,
                   descending: bool = False) -> None:
    """
    Sorts a list in-place using the selection sort algorithm.
    
    :param data: List to be sorted.
    :param comparator: Function to determine the order of elements.
    :param descending: True for descending order, False for ascending.
    """
    size = len(data)
    for i in range(size):
        min_index = i
        for j in range(i+1,size):
            if do_comparison(data[j],data[min_index],comparator,descending):
                min_index = j
        
        data[i],data[min_index] = data[min_index],data[i]


def bubble_sort(data: List[T], *, comparator: Callable[[T, T], bool] = lambda x, y: x < y,
                descending: bool = False) -> None:
    """
    Sorts a list in-place using the bubble sort algorithm.
    
    :param data: List to be sorted.
    :param comparator: Function to determine the order of elements.
    :param descending: True for descending order, False for ascending.
    """
    size = len(data)
    for i in range(size):
      for j in range(0,size-i-1):
        if not do_comparison(data[j],data[j+1],comparator,descending):
          data[j],data[j+1] = data[j+1],data[j]



def insertion_sort(data: List[T], *, comparator: Callable[[T, T], bool] = lambda x, y: x < y,
                   descending: bool = False) -> None:
    """
    Sorts a list in-place using the insertion sort algorithm.
    
    :param data: List to be sorted.
    :param comparator: Function to determine the order of elements.
    :param descending: True for descending order, False for ascending.
    """
    size = len(data)
    for i in range(size):
      for j in range(i+1,size):
        if not do_comparison(data[i],data[j],comparator,descending):
          data[i],data[j] = data[j],data[i]


def hybrid_merge_sort(data: List[T], *, threshold: int = 12,
                      comparator: Callable[[T, T], bool] = lambda x, y: x < y, descending: bool = False) -> None:
    """
    Sorts a list using a hybrid of merge sort and insertion sort algorithms.
    
    :param data: List to be sorted.
    :param threshold: Size at which to switch from merge sort to insertion sort.
    :param comparator: Function to determine the order of elements.
    :param descending: True for descending order, False for ascending.
    """

    def merge(data: List[T], start: int, mid: int, end: int):
      res = []
      i, j = start, mid + 1
      while i <= mid and j <= end:
          if do_comparison(data[i], data[j], comparator, descending):
              res.append(data[i])
              i += 1
          else:
              res.append(data[j])
              j += 1

      while i <= mid:
          res.append(data[i])
          i += 1
      while j <= end:
          res.append(data[j])
          j += 1

      for k, val in enumerate(res):
          data[start + k] = val

    def insertion_sort(data: List[T], start: int, end: int, *, comparator: Callable[[T, T], bool], descending: bool = False) -> None:
      for i in range(start + 1, end + 1):
          key = data[i]
          j = i - 1
          while j >= start and do_comparison(key, data[j], comparator, descending):
              data[j + 1] = data[j]
              j -= 1
          data[j + 1] = key
          
    def sort_and_merge(start: int, end: int):
      if end - start <= threshold:
          insertion_sort(data, start, end, comparator=comparator, descending=descending)
          return

      if start < end:
          mid = (start + end) // 2
          sort_and_merge(start, mid)
          sort_and_merge(mid + 1, end)
          merge(data, start, mid, end)
  
    sort_and_merge(0, len(data) - 1)
  

def quicksort(data: List[T]) -> None:
    """
    Sorts a list in place using quicksort
    :param data: Data to sort
    """

    def quicksort_inner(first: int, last: int) -> None:
        """
        Sorts portion of list at indices in interval [first, last] using quicksort

        :param first: first index of portion of data to sort
        :param last: last index of portion of data to sort
        """
        # List must already be sorted in this case
        if first >= last:
            return

        left = first
        right = last

        # Need to start by getting median of 3 to use for pivot
        # We can do this by sorting the first, middle, and last elements
        midpoint = (right - left) // 2 + left
        if data[left] > data[right]:
            data[left], data[right] = data[right], data[left]
        if data[left] > data[midpoint]:
            data[left], data[midpoint] = data[midpoint], data[left]
        if data[midpoint] > data[right]:
            data[midpoint], data[right] = data[right], data[midpoint]
        # data[midpoint] now contains the median of first, last, and middle elements
        pivot = data[midpoint]
        # First and last elements are already on right side of pivot since they are sorted
        left += 1
        right -= 1

        # Move pointers until they cross
        while left <= right:
            # Move left and right pointers until they cross or reach values which could be swapped
            # Anything < pivot must move to left side, anything > pivot must move to right side
            #
            # Not allowing one pointer to stop moving when it reached the pivot (data[left/right] == pivot)
            # could cause one pointer to move all the way to one side in the pathological case of the pivot being
            # the min or max element, leading to infinitely calling the inner function on the same indices without
            # ever swapping
            while left <= right and data[left] < pivot:
                left += 1
            while left <= right and data[right] > pivot:
                right -= 1

            # Swap, but only if pointers haven't crossed
            if left <= right:
                data[left], data[right] = data[right], data[left]
                left += 1
                right -= 1

        quicksort_inner(first, left - 1)
        quicksort_inner(left, last)

    # Perform sort in the inner function
    quicksort_inner(0, len(data) - 1)


###########################################################
# DO NOT MODIFY
###########################################################

class Score:
    """
    Class that represents SAT scores
    NOTE: While it is possible to implement Python "magic methods" to prevent the need of a key function,
    this is not allowed for this application problems so students can learn how to create comparators of custom objects.
    Additionally, an individual section score can be outside the range [400, 800] and may not be a multiple of 10
    """

    __slots__ = ['english', 'math']

    def __init__(self, english: int, math: int) -> None:
        """
        Constructor for the Score class
        :param english: Score for the english portion of the exam
        :param math: Score for the math portion of the exam
        :return: None
        """
        self.english = english
        self.math = math

    def __repr__(self) -> str:
        """
        Represent the Score as a string
        :return: representation of the score
        """
        return str(self)

    def __str__(self) -> str:
        """
        Convert the Score to a string
        :return: string representation of the score
        """
        return f'<English: {self.english}, Math: {self.math}>'


###########################################################
# MODIFY BELOW
###########################################################

def better_than_most(scores: List[Score], student_score: Score) -> str:
    """
    Evaluates if a student's SAT scores exceed the median scores of a list.

    :param scores: List of students' SAT scores.
    :param student_score: The student's SAT scores to compare.
    :return: 'Both', 'English', 'Math', or 'None', indicating which scores are above the median.
    """
    
    def median(sorted_scores: List[int]) -> float:
            n = len(sorted_scores)
            mid = n // 2
            if not sorted_scores: #checks if the list of scores is empty
              return 0
            if n % 2 == 0: #if it is even number of elements
                return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
            else:
                return sorted_scores[mid]

    english_scores = []
    math_scores = []

    #appending scores into the created lists
    for score in scores:
      english_scores.append(score.english)
      math_scores.append(score.math)

    #sorting the scores in the two lists
    hybrid_merge_sort(english_scores)
    hybrid_merge_sort(math_scores)
    
    #finding the median in the two lists
    median_english = median(english_scores)
    median_math = median(math_scores)

    #checking is students score is greater than median and addes it to resultant list 
    res = []
    if student_score.english > median_english:
      res.append('English')
    if student_score.math > median_math:
      res.append('Math')
    
    #based on what is in the resultant list returns output
    if not res:
      return 'None'
    elif 'English' in res and 'Math' in res:
      return 'Both'
    elif 'English' in res:
      return 'English'
    else:
      return 'Math'