"""
Project 5
CSE 331 SS24
Authors: Hank Murdock, Joel Nataren, Aaron Elkin, Divyalakshmi Varadha, Ethan Cook
starter.py
"""
import math
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue
import heapq

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculate the height of a node within a binary search tree (BST).

        The height of a node is the number of edges on the longest path from the node to a leaf.
        A leaf node's height is 0, and a non-existent node (`None`) has a height of -1.

        Parameters:
        - root (Node): The node from which to calculate the height.

        Returns:
        int: The height of the given node.
        """
        if root is None:
            return -1
        return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Insert a new value into the binary search tree (BST).

        This method initiates the insertion by calling a recursive helper function
        to correctly place the new value according to BST rules. If the tree is empty,
        the new value becomes the root. The tree's size is incremented upon a successful insertion.

        Parameters:
        - root (Node): The current root of the BST (ignored in this implementation as `self.origin` is always used).
        - val (T): The value to be inserted into the BST.

        Returns:
        None
        """
        self.origin = self._insert(self.origin, val)

    def _insert(self, current, val):
        if current is None:
            self.size += 1
            return Node(val)
        elif val < current.value:
            current.left = self._insert(current.left, val)
            current.left.parent = current
        elif val > current.value:
            current.right = self._insert(current.right, val)
            current.right.parent = current
        
        current.height = 1 + max(self.height(current.left), self.height(current.right))
        return current


    def _find_max(self, node):
        while node.right is not None:
            node = node.right
        return node

    def _remove(self, node, val):
        if node is None:
            return None
        if val < node.value:
            node.left = self._remove(node.left, val)
        elif val > node.value:
            node.right = self._remove(node.right, val)
        else:
            if node.left is None:
                temp = node.right
                if temp:
                    temp.parent = node.parent  
                if node == self.origin:
                    self.origin = temp  
                node = None
                self.size -= 1
                return temp
            elif node.right is None:
                temp = node.left
                if temp:
                    temp.parent = node.parent  
                if node == self.origin:
                    self.origin = temp  
                node = None
                self.size -= 1
                return temp
            else:
                temp = self._find_max(node.left)
                node.value = temp.value
                node.left = self._remove(node.left, temp.value)

        if node:
            node.height = 1 + max(self.height(node.left), self.height(node.right))
        return node


    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Remove a value from the BST.

        Initiates the removal process by calling a recursive helper function. This method
        ensures that the BST remains valid after the removal. If the tree is empty or the
        value is not found, the tree remains unchanged.

        Parameters:
        - root (Node): The current root of the BST (ignored in this implementation as `self.origin` is always used).
        - val (T): The value to be removed from the BST.

        Returns:
        Optional[Node]: The root of the updated BST, which may be unchanged if the value was not found.
        """
        self.origin = self._remove(self.origin, val)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Search for a value in the BST.

        Recursively navigates through the BST to find a node containing the given value.
        Returns the first node matching the value if it exists, otherwise, returns `None`.

        Parameters:
        - root (Node): The root node from which to start the search.
        - val (T): The value to search for in the BST.

        Returns:
        Optional[Node]: The node containing the searched value, or `None` if the value is not found.
        """
        if root is None or root.value == val:
            return root
        if val < root.value:
            return self.search(root.left, val) or root
        return self.search(root.right, val) or root



class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in an AVL tree
        :param root: The root node of the subtree for which the height is being calculated.
        :return: The height of the subtree rooted at the given node. Returns -1 if the subtree is empty.
        """
        return -1 if root is None else root.height

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        Performs a left rotation on the subtree rooted at the specified root node.
        :param root: The root node of the subtree to be rotated.
        :return: The new root node of the subtree after the rotation.
        """
        if root is None: 
            return None
        new_root = root.right
        root.right = new_root.left
        if new_root.left is not None:
            new_root.left.parent = root
        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root == root.parent.left:
            root.parent.left = new_root
        else:
            root.parent.right = new_root
        new_root.left = root
        root.parent = new_root
        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        Performs a right rotation on the subtree rooted at the specified root node
        :param root: The root node of the subtree to be rotated.
        :return: The new root node of the subtree after the rotation.
        """
        if root is None: 
            return None
        new_root = root.left
        root.left = new_root.right
        if new_root.right is not None:
            new_root.right.parent = root
        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root == root.parent.right:
            root.parent.right = new_root
        else:
            root.parent.left = new_root
        new_root.right = root
        root.parent = new_root
        
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Computes the balance factor of a node in an AVL tree
        :param root: The root node of the subtree for which to compute the balance factor.
        :return: The balance factor of the subtree rooted at the given node.
        """
        if root is None:
            return 0  
        return self.height(root.left) - self.height(root.right)

    def update_height(self, node: Node) -> None:
        """
        Helper function that updates the height of a node based on the heights of its children.
        :param node: a node in the tree
        """
        node.height = 1 + max(
            node.left.height if node.left else -1, 
            node.right.height if node.right else -1
        )
    
    def rebalance(self, node: Node) -> Optional[Node]:
        """
        Performs a rebalance rooted at the specified node
        :param node: The root node of the subtree to be rebalanced.
        :return: The new root node of the subtree after the rebalance.
        """
        if not node:
            return None
        self.update_height(node)

        balance = self.balance_factor(node)

        if balance > 1:
            if self.balance_factor(node.left) < 0:
                node.left = self.left_rotate(node.left)
            new_root = self.right_rotate(node)
            
        elif balance < -1:
            if self.balance_factor(node.right) > 0:
                node.right = self.right_rotate(node.right)
            new_root = self.left_rotate(node)

        else:
            new_root = node
        if new_root.left:
            self.update_height(new_root.left)
        if new_root.right:
            self.update_height(new_root.right)
        self.update_height(new_root)
        return new_root

    def insert(self, root: Node, val: T, parent: Optional[Node] = None) -> Optional[Node]:
        """
        Inserts a new node with the given value into the AVL tree and rebalances the tree.
        :param root: The current root node of the AVL tree or a subtree.
        :param val: The value to be inserted.
        :return: The new root of the subtree after insertion and rebalancing.
        """
        if root is None:
            self.size += 1
            new_node = Node(val, parent)
            if self.origin is None:
                self.origin = new_node
            return new_node
        
        if val < root.value:
            root.left = self.insert(root.left, val, root)
        elif val > root.value:
            root.right = self.insert(root.right, val, root)
        else:
            return root
        return self.rebalance(root)

    def remove(self, root: Optional[Node], val: T) -> Optional[Node]:
        """
        Removes a node with the specified value from the AVL tree starting at the given root node. 
        :param root: The root node of the subtree from which the node is to be removed.
        :param val: The value of the node to be removed.
        :return: The new root of the subtree after the removal and rebalancing.
        """
        if root is None:
            return None
        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            if root.left is None or root.right is None:
                temp = root.left if root.left else root.right
                if temp:
                    temp.parent = root.parent
                if root is self.origin:
                    self.origin = temp
                self.size -= 1
                return temp
            else:
                temp = self.max(root.left)
                root.value = temp.value
                root.left = self.remove(root.left, temp.value)
                
                if root.left:
                    root.left.parent = root

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return self.rebalance(root)

    def find_max(self, node: Optional[Node]) -> Optional[Node]:
        """
        Finds the node with the maximum value in the given subtree.
        This helper function is used during the removal of a node with two children to find the in-order predecessor.
        :param node: The root node of the subtree to search.
        :return: The node with the maximum value in the subtree.
        """
        current = node
        while current.right is not None:
            current = current.right
        return current

    def min(self, root: Optional[Node]) -> Optional[Node]:
        """
        Finds the node with the minimum value within the subtree rooted at the given node
        :param root: The root node of the subtree from which to find the minimum value.
        :return: The node containing the smallest value within the subtree, or None if the subtree is empty.
        """
        if root is None or root.left is None:
            return root
        return self.min(root.left)

    def max(self, root: Optional[Node]) -> Optional[Node]:
        """
        Searches for and returns the node with the largest value within the subtree rooted at the specified node.
        :param root: The root of the subtree within which to search for the maximum value.
        :return: The node with the largest value within the subtree, or None if the subtree is empty.
        """
        if root is None or root.right is None:
            return root
        return self.max(root.right)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for a node with a specified value in the subtree rooted at the given node
        :param root: The root node of the subtree in which the search is to be performed.
        :param val: The value to search for.
        :return: The node containing the specified value if it exists, otherwise the node under which the value would be inserted.
        """
        if root is None or root.value == val:
            return root
        elif val < root.value:
            return self.search(root.left, val) if root.left else root
        else:
            return self.search(root.right, val) if root.right else root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Generates the nodes of the subtree rooted at the specified node using an inorder traversal method
        :param root: The root node of the subtree to traverse.
        :return: A generator yielding the nodes in the subtree in inorder sequence.
        """
        if root is not None:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        Makes the AVL tree class iterable using an inorder traversal
        :return: A generator yielding the nodes of the tree in inorder.
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a preorder traversal of the AVL tree, yielding nodes in a "root, left, right" order.
        :param root: The root node of the subtree to begin traversal from.
        :return: A generator yielding nodes in preorder.
        """
        if root:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a postorder traversal of the AVL tree, yielding nodes in a "left, right, root" order.
        :param root: The root node of the subtree to begin traversal from.
        :return: A generator yielding nodes in postorder.
        """
        if root:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root
            
    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a level-order (breadth-first) traversal of the AVL tree, yielding nodes level by level from top to bottom.
        :param root: The root node of the subtree to begin traversal from.
        :return: A generator yielding nodes in level-order.
        """
        if not root:
            return
        queue = deque([root])
        while queue:
            node = queue.popleft()
            yield node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)


####################################################################################################

class User:
    """
    Class representing a user of the stock marker.
    Note: A user can be both a buyer and seller.
    """

    def __init__(self, name, pe_ratio_threshold, div_yield_threshold):
        self.name = name
        self.pe_ratio_threshold = pe_ratio_threshold
        self.div_yield_threshold = div_yield_threshold


####################################################################################################

class Stock:
    __slots__ = ['ticker', 'name', 'price', 'pe', 'mkt_cap', 'div_yield']
    TOLERANCE = 0.001

    def __init__(self, ticker, name, price, pe, mkt_cap, div_yield):
        """
        Initialize a stock.

        :param name: Name of the stock.
        :param price: Selling price of stock.
        :param pe: Price to earnings ratio of the stock.
        :param mkt_cap: Market capacity.
        :param div_yield: Dividend yield for the stock.
        """
        self.ticker = ticker
        self.name = name
        self.price = price
        self.pe = pe
        self.mkt_cap = mkt_cap
        self.div_yield = div_yield

    def __repr__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return f"{self.ticker}: PE: {self.pe}"

    def __str__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return repr(self)

    def __lt__(self, other):
        """
        Check if the stock is less than the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is less than the other stock, False otherwise.
        """
        return self.pe < other.pe

    def __eq__(self, other):
        """
        Check if the stock is equal to the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is equal to the other stock, False otherwise.
        """
        return abs(self.pe - other.pe) < self.TOLERANCE


def make_stock_from_dictionary(stock_dictionary: dict[str: str]) -> Stock:
    """
    Builds an AVL tree with the given stock dictionary.

    :param stock_dictionary: Dictionary of stocks to be inserted into the AVL tree.
    :return: A stock in a Stock object.
    """
    stock = Stock(stock_dictionary['ticker'], stock_dictionary['name'], stock_dictionary['price'], \
                  stock_dictionary['pe_ratio'], stock_dictionary['market_cap'], stock_dictionary['div_yield'])
    return stock


def build_tree_with_stocks(stocks_list: List[dict[str: str]]) -> AVLTree:
    """
    Builds an AVL tree with the given list of stocks.

    :param stocks_list: List of stocks to be inserted into the AVL tree.
    :return: AVL tree with the given stocks.
    """
    avl = AVLTree()
    for stock in stocks_list:
        stock = make_stock_from_dictionary(stock)
        avl.insert(avl.origin, stock)
    return avl


####################################################################################################
# Implement functions below this line. #
####################################################################################################

def recommend_stock(stock_tree: AVLTree, user: User, action: str) -> Optional[Stock]:
    """
    The recommend_stock function is designed to aid investors (users) in making informed decisions by recommending best
    stock to buy or sell based on specified criteria. Utilizing the AVL Tree data structure, this function shifts
    through stocks to find the one that best matches the investor's financial thresholds and goals.
    :param stock_tree: AVL Tree containing stock nodes.
    :param user: A user object representing the investor's preferences. It includes attributes such as name,
                 pe_ratio_threshold, and div_yield_threshold to guide the stock recommendation process.
    :param action: A string indicating the desired action, either 'buy' or 'sell'.
                   This determines the criteria used to filter and recommend stocks.
    """
    if stock_tree is None or stock_tree.size == 0:
        return None

    best_stock = None

    for node in stock_tree:
        stock = node.value
        if action == 'buy':
            if stock.pe < user.pe_ratio_threshold and stock.div_yield > user.div_yield_threshold:
                if best_stock is None or (stock.pe < best_stock.pe and stock.div_yield > best_stock.div_yield):
                    best_stock = stock
        elif action == 'sell':
            if stock.pe > user.pe_ratio_threshold or stock.div_yield < user.div_yield_threshold:
                if best_stock is None or (stock.pe > best_stock.pe and stock.div_yield < best_stock.div_yield):
                    best_stock = stock

    return best_stock

def prune(stock_tree: AVLTree, threshold: float = 0.05) -> None:
    """
    This is the primary function for this application problem

    :param stock_tree: The AVL Tree to be pruned.
    :param threshold: Any subtree with all pe values less than this gets removed.
    TIP: You may need to create helper functions for this problem.
    """
    def remove(node: Node):
        if node is None:
            return None

        if node.left is not None:
            node.left = remove(node.left)
        if node.right is not None:
            node.right = remove(node.right)

        if node.value.pe < threshold:
            if node.left is None and node.right is None:
                return None
            elif node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                return node

        return node

    stock_tree.origin = remove(stock_tree.origin)


####################################################################################################
####################### EXTRA CREDIT ##############################################################
####################################################################################################

class Blackbox:
    def __init__(self):
        """
        Initialize a minheap.
        """
        self.heap = []

    def store(self, value: T):
        """
        Push a value into the heap while maintaining minheap property.

        :param value: The value to be added.
        """
        heapq.heappush(self.heap, value)

    def get_next(self) -> T:
        """
        Pop minimum from min heap.

        :return: Smallest value in heap.
        """
        return heapq.heappop(self.heap)

    def __len__(self):
        """
        Length of the heap.

        :return: The length of the heap
        """
        return len(self.heap)

    def __repr__(self) -> str:
        """
        The string representation of the heap.

        :return: The string representation of the heap.
        """
        return repr(self.heap)

    __str__ = __repr__


class HuffmanNode:
    __slots__ = ['character', 'frequency', 'left', 'right', 'parent']

    def __init__(self, character, frequency):
        self.character = character
        self.frequency = frequency

        self.left = None
        self.right = None
        self.parent = None

    def __lt__(self, other):
        """
        Checks if node is less than other.

        :param other: The other node to compare to.
        """
        return self.frequency < other.frequency

    def __repr__(self):
        """
        Returns string representation.

        :return: The string representation.
        """
        return '<Char: {}, Freq: {}>'.format(self.character, self.frequency)

    __str__ = __repr__


class HuffmanTree:
    __slots__ = ['root', 'blackbox']

    def __init__(self):
        self.root = None
        self.blackbox = Blackbox()

    def __repr__(self):
        """
        Returns the string representation.

        :return: The string representation.
        """
        if self.root is None:
            return "Empty Tree"

        lines = pretty_print_binary_tree(self.root, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    __str__ = __repr__

    def make_char_map(self) -> dict[str: str]:
        """
        Create a binary mapping from the huffman tree.

        :return: Dictionary mapping from characters to "binary" strings.
        """
        mapping = {}

        def traversal(root: HuffmanNode, current_str: str):
            if not root:
                return

            if not root.left and not root.right:
                mapping[root.character] = current_str
                return

            if root.left:
                traversal(root.left, current_str=current_str + '0')

            if root.right:
                traversal(root.right, current_str=current_str + '1')

        traversal(self.root, '')

        return mapping

    def compress(self, input: str) -> tuple[dict[str: str], List[str]]:
        """
        Compress the input data by creating a map via huffman tree.

        :param input: String to compress.
        :return: First value to return is the mapping from characters to binary strings.
        Second value is the compressed data.
        """
        self.build(input)

        mapping = self.make_char_map()

        compressed_data = []

        for char in input:
            compressed_data.append(mapping[char])

        return mapping, compressed_data

    def decompress(self, mapping: dict[str: str], compressed: List[str]) -> str:
        """
        Use the mapping from characters to binary strings to decompress the array of bits.

        :param mapping: Mapping of characters to binary strings.
        :param compressed: Array of binary strings that are encoded.
        """

        reverse_mapping = {v: k for k, v in mapping.items()}

        decompressed = ""

        for encoded in compressed:
            decompressed += reverse_mapping[encoded]

        return decompressed

    ########################################################################################
    # Implement functions below this line. #
    ########################################################################################

    def build(self, chars: str) -> None:
        """
        Given some input construct a Huffman tree based off that input.
        
        :param chars: A string to create a Huffman tree based off of. These are the characters to calculate your
        frequencies from.
        """
        if len(chars) == 0:
            return

        freq_map = {}
        for char in chars:
            if char in freq_map:
                freq_map[char] += 1
            else:
                freq_map[char] = 1

        for char, freq in freq_map.items():
            self.blackbox.store(HuffmanNode(char, freq))

        while len(self.blackbox) > 1:
            left = self.blackbox.get_next()
            right = self.blackbox.get_next()

            parent = HuffmanNode(None, left.frequency + right.frequency)
            parent.left = left
            parent.right = right

            left.parent = parent
            right.parent = parent

            self.blackbox.store(parent)

        self.root = self.blackbox.get_next()


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root) == HuffmanNode:
            node_repr = repr(root)
        elif type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    line1.append(node_repr)
    line2.append(" " * new_root_width)


    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1


    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    return new_box, len(new_box[0]), new_root_start, new_root_end


if __name__ == "__main__":
    pass
