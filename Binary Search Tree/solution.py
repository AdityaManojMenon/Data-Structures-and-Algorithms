from collections import deque

class Node:
    # DO NOT MODIFY THIS CLASS #
    __slots__ = 'value', 'parent', 'left', 'right', 'height'

    def __init__(self, value, parent=None, left=None, right=None):
        """
        Initialization of a node
        :param value: value stored at the node
        :param parent: the parent node
        :param left: the left child node
        :param right: the right child node
        """
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right
        self.height = 0

    def __eq__(self, other):
        """
        Determine if the two nodes are equal
        :param other: the node being compared to
        :return: true if the nodes are equal, false otherwise
        """
        if type(self) is not type(other):
            return False
        return self.value == other.value

    def __str__(self):
        """String representation of a node by its value"""
        return str(self.value)

    def __repr__(self):
        """String representation of a node by its value"""
        return str(self.value)


class BSTTree:

    def __init__(self):
        # DO NOT MODIFY THIS FUNCTION #
        """
        Initializes an empty Binary Search Tree
        """
        self.root = None
        self.size = 0
        self.answer = 0

    def __eq__(self, other):
        """
        Describe equality comparison for BSTs ('==')
        :param other: BST being compared to
        :return: True if equal, False if not equal
        """
        if self.size != other.size:
            return False
        if self.root != other.root:
            return False
        if self.root is None or other.root is None:
            return True  # Both must be None

        if self.root.left is not None and other.root.left is not None:
            r1 = self._compare(self.root.left, other.root.left)
        else:
            r1 = (self.root.left == other.root.left)
        if self.root.right is not None and other.root.right is not None:
            r2 = self._compare(self.root.right, other.root.right)
        else:
            r2 = (self.root.right == other.root.right)

        result = r1 and r2
        return result

    def _compare(self, t1, t2):
        """
        Recursively compares two trees, used in __eq__.
        :param t1: root node of first tree
        :param t2: root node of second tree
        :return: True if equal, False if nott
        """
        if t1 is None or t2 is None:
            return t1 == t2
        if t1 != t2:
            return False
        result = self._compare(t1.left, t2.left) and self._compare(t1.right, t2.right)
        return result

    def insert(self, node, value) -> None:
        """
        Inserts a node with a value into the BST - don't do anything if value is in BST already
        :param node: the root of the subtree we are traversing
        :param value: the value to insert into the BST
        """

        if node is None:
            self.root=Node(value)
            self.size += 1
        else:
            if node.value==value:
                return
            elif value > node.value:
                if node.right is None:
                    node.right = Node(value,parent=node)
                    self.size +=1
                else:
                    self.insert(node.right,value)
            else:
                if node.left is None:
                    node.left=Node(value, parent=node)
                    self.size +=1
                else:
                    self.insert(node.left,value)

    # Application Problem
    # Using Depth First Search
    def range_sum1(self, root, low, high) -> int:
        def dfs(node):
            if node:
                if low<=node.value<=high:
                    self.answer += node.value
                if low <node.value:
                    dfs(node.left)
                if node.value<high:
                    dfs(node.right)

        dfs(root)
        return self.answer

    # Breadth First Search
    def range_sum2(self, root, low, high) -> int:
        node_list = [root]
        while node_list:
            node = node_list.pop()
            if node:
                if low<=node.value<=high:
                    self.answer +=node.value
                if low<node.value:
                    node_list.append(node.left)
                if node.value<high:
                    node_list.append(node.right)
        return self.answer

    def range_sum3(self, root, low, high) -> int:
        rs_deque = deque()
        rs_deque.append(root)
        while rs_deque:
            curr_node = rs_deque.popleft()
            if curr_node:
                if low<=curr_node.value<=high:
                    self.answer +=curr_node.value
                if low<curr_node.value:
                    rs_deque.append(curr_node.left)
                if curr_node.value<high:
                    rs_deque.append(curr_node.right)
        return self.answer


# Another Application Problem
    def find_closest(self, root, target) -> int:
        # Initialize the closest value to the root's value (assuming the tree is not empty)
        if root is None:
            return float('inf')  # Return infinity if the tree is empty
        return self.find_closest_helper(root, target, root.value)

    def find_closest_helper(self, node, target, closest) -> int:
        # Base case: node is None
        if node is None:
            return closest
        # Update closest if the current node is closer to the target
        if abs(target - node.value) < abs(target - closest):
            closest = node.value
        # Decide whether to go left or right
        if target < node.value:
            return self.find_closest_helper(node.left, target, closest)
        elif target > node.value:
            return self.find_closest_helper(node.right, target, closest)
        else:  # target == node.value
            return node.value  # Found the exact target

    def find_closest_iter(self, root, target) -> int:
        closest = float('inf')
        current_node = root
        while current_node is not None:
            # Update closest if the current node is closer
            if abs(target - current_node.value) < abs(target - closest):
                closest = current_node.value
            # Move to the next node
            if target < current_node.value:
                current_node = current_node.left
            elif target > current_node.value:
                current_node = current_node.right
            else:  # Found the exact target
                return current_node.value
        return closest
