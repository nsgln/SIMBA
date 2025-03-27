"""This file contains the class needed to represent a node.

@Author: Nina Singlan."""


class Node:
    """This class represents a node in a graph.

    Attributes:
        identifier: (int) A unique identifier for the node.
        value: (float) The value of the node.
        neighbors: (Set[Node]) The neighbors of the node.
    """

    """Initializes the node with the given identifier and value.

    Parameters:
        identifier: (int) A unique identifier for the node.
        value: (float) The value of the node."""

    def __init__(self, identifier: int, value: float):
        self.identifier = identifier
        self.value = value
        self.neighbors = set()

    """Represents the node as a string.

    Returns:
        (str) A string representation of the node."""

    def __str__(self) -> str:
        return f"Node {self.identifier}: {self.value}"

    """Checks if the node is equal to another node.

    Parameters:
        other: (object) The object to compare to.

    Returns:
        (bool) True if the nodes are equal, False otherwise"""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.identifier == other.identifier and self.value == other.value

    """Returns the hash value of the node.

    Returns:
        (int) The hash value of the node."""

    def __hash__(self) -> int:
        return hash(self.identifier)

    """Adds a neighbor to the node.
    
    Parameters:
        neighbor: (Node) The neighbor to add.
        
    Raises:
        ValueError: If the neighbor is already a neighbor of the node.
        TypeError: If the neighbor is not a Node object."""

    def add_neighbor(self, neighbor: object) -> None:
        if not isinstance(neighbor, Node):
            raise TypeError("The neighbor should be a Node object.")
        if neighbor in self.neighbors:
            raise ValueError("The neighbor is already a neighbor of the node.")
        self.neighbors.add(neighbor)
