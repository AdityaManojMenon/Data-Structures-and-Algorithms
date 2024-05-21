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
        INSERT DOCSTRING HERE
        """
        pass

    def insert(self, root: Node, val: T) -> None:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass


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
        INSERT DOCSTRING HERE
        """
        pass

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def balance_factor(self, root: Node) -> int:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def min(self, root: Node) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def max(self, root: Node) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def __iter__(self) -> Generator[Node, None, None]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        INSERT DOCSTRING HERE
        """
        pass

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        INSERT DOCSTRING HERE
        """
        pass


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
    INSERT DOCSTRING HERE
    """
    pass


def prune(stock_tree: AVLTree, threshold: float = 0.05) -> None:
    """
    INSERT DOCSTRING HERE
    """
    pass


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
        INSERT DOCSTRING HERE
        """
        pass


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

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
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

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


if __name__ == "__main__":
    pass
