class Recursion:
    def __init__(self):
        # initialize memo
        self.memo = {}

    def bad_fib(self, n):
        """
        Find the nth Fibonacci number using recursion
        :param n: the index of the Fibonacci number to find
        :return: the nth Fibonacci number
        """
        # Base case: if n is 0, the nth Fibonacci number is 0
        if n == 0:
            return 0
        # Base case: if n is 1, the nth Fibonacci number is 1
        elif n == 1:
            return 1
        # Recursive case: if n is greater than 1, the nth Fibonacci number is
        # the sum of the (n-1)th and (n-2)th Fibonacci numbers
        else:
            return self.bad_fib(n - 1) + self.bad_fib(n - 2)

    def good_fib(self, n: int) -> int:
        # Base case: if n is less than 2, the fibonacci number is n
        if n < 2:
            return n
        # Check if fib(n) has already been calculated and stored in memo
        if n in self.memo:
        # If it has, return the stored value
            return self.memo[n]
        else:
        # If it hasn't, calculate fib(n) by recursively finding fib(n-1) and fib(n-2)
            res = self.good_fib(n - 1) + self.good_fib(n - 2)
        # Store the calculated value of fib(n) in memo for future use
            self.memo[n] = res
        return res

    def nested_list_product_sum(self, data, depth=1):
        # Initialize sum to keep track of the total sum of all elements
        sum = 0 
        # Iterate through each element in the input array
        for element in data:
        # Check if the current element is a list (nested list)
            if isinstance(element, list):
        # If it is a nested list, call the function recursively on that list
        # and add the returned value to the sum
        # Also, increment the depth by 1 for each level of nesting
                sum += self.nested_list_product_sum(element, depth + 1)
            else:
        # If the element is not a list, add it to the sum
                sum += element 
        # Return the final sum multiplied by the depth (to account for nested levels)
        return sum * depth

