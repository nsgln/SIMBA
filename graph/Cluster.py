""" This file contains the class needed to represent a cluster.

@Author: Nina Singlan."""
from typing import List, Tuple, Dict

import numpy as np

from graph.Node import Node


class Cluster:
    """This class represents a cluster in a graph.

    Attributes:
        identifier: (int) A unique identifier for the cluster.
        nodes: (List[Node]) The nodes in the cluster.
        edges: (Dict[Tuple[Node, Node], float | Tuple[float, int, int | None]]) The edges between the nodes in
                the cluster.
        real: (bool) Indicates if it's a real cluster.
        number_edges: (int) The number of edges in the cluster.
        number_nodes: (int) The number of nodes in the cluster.
        score: (float) The score of the cluster.
        maximum_edges: (int) The maximum number of edges that could be in the cluster.
        acceptable_score: (float | None) The acceptable score of the cluster.
        acceptable_ratio: (float | None) The acceptable ratio of the cluster.
    """
    identifier = 0

    """Initializes the cluster with the given nodes.

    Parameters :
        nodes: (List[Node]) The nodes in the cluster.
        edges: (Dict[Tuple[Node, Node], float | Tuple[float, int, int | None]]) The edges between the nodes in the cluster.
        avoid_isolated_nodes: (bool) Whether to avoid isolated nodes in the cluster.
            Optional, default is True."""

    def __init__(self, nodes: List[Node], edges: Dict[Tuple[Node, Node], float | Tuple[float, int, int | None]],
                 avoid_isolated_nodes: bool = True):
        self.identifier = Cluster.identifier
        Cluster.identifier += 1
        # Check for duplicate nodes
        nodes = list(set(nodes))
        if avoid_isolated_nodes:
            nodes, edges = self._check_isolated_nodes(nodes, edges)
        self.nodes = nodes
        self.edges = edges
        if len(edges.values()) == 0:
            self.real = True
        elif isinstance(list(edges.values())[0], (float, np.floating)):
            self.real = True
        else:
            self.real = False
        if self.real:
            self.number_edges = len(edges)
            self.number_nodes = len(nodes)
        else:
            self.number_nodes = np.sum([val[2] for val in self.edges.values() if val[2] is not None])
            self.number_edges = np.sum([val[1] for val in self.edges.values()])
        self.maximum_edges = (self.number_nodes * (self.number_nodes - 1)) / 2
        self.acceptable_score = None
        self.acceptable_ratio = None
        self.score = self._compute_score()

    """Represents the cluster as a string.

    Returns:
        (str) A string representation of the cluster."""

    def __str__(self) -> str:
        return f"Cluster {self.identifier} with {self.number_nodes} nodes, {self.number_edges} edges, and a score of {self.score}."

    """Checks if the cluster is equal to another cluster.

    Parameters:
        other: (object) The object to compare to.

    Returns:
        (bool) True if the clusters are equal, False otherwise."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cluster):
            return False
        res = True
        for node in self.nodes:
            if node not in other.nodes:
                res = False
                break
        for node in other.nodes:
            if node not in self.nodes:
                res = False
                break
        return res

    """Checks if the cluster contains a node.

    Parameters:
        item: (Node) The node to check for.

    Returns:
        (bool) True if the cluster contains the node, False otherwise"""

    def __contains__(self, item: Node) -> bool:
        return item in self.nodes

    """Function to get the size of the cluster.

    Returns:
        (int) The size of the cluster."""

    def __len__(self) -> int:
        return self.number_nodes

    """Returns the hash value of the cluster.
    
    Returns:
        (int) The hash value of the cluster."""

    def __hash__(self):
        return hash(self.identifier)

    """Computes the score of the cluster.

    Returns:
        (float) The score of the cluster."""

    def _compute_score(self) -> float:
        if len(self.edges) < 1 or len(self.nodes) < 2:
            return 0
        if self.real:
            weighting_sum = np.sum(list(self.edges.values()))
        else:
            weighting_sum = np.sum([val[0] for val in self.edges.values()])
        return weighting_sum * (self.number_edges / self.maximum_edges)

    """Function to remove a node from the cluster.

    Parameters:
        node: (Node) The node to remove.
        avoid_isolated_nodes: (bool) Whether to avoid isolated nodes in the cluster.
            Optional, default is True.

    Returns:
        (Cluster) The cluster without the node."""

    def remove_node(self, node: Node, avoid_isolated_nodes: bool = True) -> 'Cluster':
        new_nodes = [n for n in self.nodes if n != node]
        new_edges = {}
        for node in new_nodes:
            for node2 in new_nodes:
                if (node, node2) in self.edges:
                    new_edges[(node, node2)] = self.edges[(node, node2)]
                elif (node2, node) in self.edges:
                    new_edges[(node2, node)] = self.edges[(node2, node)]

        return Cluster(new_nodes, new_edges, avoid_isolated_nodes)

    """Function to add a node to the cluster.

    Parameters:
        node: (Node) The node to add.
        edges: (Dict[Tuple[Node, Node], float]) The edges of the graph.
        avoid_isolated_nodes: (bool) Whether to avoid isolated nodes in the cluster.
            Optional, default is True.

    Returns:
        (Cluster) The cluster with the added node."""

    def add_node(self, node: Node, edges: Dict[Tuple[Node, Node], float], avoid_isolated_nodes: bool = True
                 ) -> 'Cluster':
        new_nodes = self.nodes + [node]
        new_edges = {edge: value for edge, value in self.edges.items()}

        if (node, node) in edges:
            new_edges[(node, node)] = edges[(node, node)]
        for node2 in self.nodes:
            if (node, node2) in edges:
                new_edges[(node, node2)] = edges[(node, node2)]
            elif (node2, node) in edges:
                new_edges[(node2, node)] = edges[(node2, node)]

        return Cluster(new_nodes, new_edges, avoid_isolated_nodes)

    """Function to get the adjacency matrix of the cluster, and the index map to retrieve the nodes in the matrix.

    Returns:
        (List[List[float]]) The adjacency matrix of the cluster.
        (Dict[Node, int]) The index map to retrieve the nodes in the matrix."""

    def get_adjacency_matrix(self) -> Tuple[List[List[float]], Dict[Node, int]]:
        adjacency_matrix = [[0 for _ in range(self.number_nodes)] for _ in range(self.number_nodes)]
        index_map = {}
        for edge, value in self.edges.items():
            node1 = self.nodes.index(edge[0])
            node2 = self.nodes.index(edge[1])
            if edge[0] not in index_map:
                index_map[edge[0]] = node1
            if edge[1] not in index_map:
                index_map[edge[1]] = node2
            adjacency_matrix[node1][node2] = value
            adjacency_matrix[node2][node1] = value
        return adjacency_matrix, index_map

    """Function to check if there are isolated nodes in the cluster.

    Parameters:
        nodes: (List[Node]) The nodes in the cluster.
        edges: (Dict[Tuple[Node, Node], float]) The edges of the cluster.

    Returns:
        (List[Node]) The nodes in the cluster without the isolated nodes.
        (Dict[Tuple[Node, Node], float]) The edges of the cluster without the edges connected to isolated nodes."""

    @staticmethod
    def _check_isolated_nodes(nodes: List[Node], edges: Dict[Tuple[Node, Node], float]) -> Tuple[List[Node],
    Dict[Tuple[Node, Node], float]]:
        isolated_nodes = [node for node in nodes
                          if node not in [edge[0] for edge in edges] and node not in [edge[1] for edge in edges]]
        new_nodes = [node for node in nodes if node not in isolated_nodes]
        new_edges = edges
        for node1 in new_nodes:
            for node2 in isolated_nodes:
                if (node1, node2) in new_edges:
                    del new_edges[(node1, node2)]
                if (node2, node1) in new_edges:
                    del new_edges[(node2, node1)]

        return new_nodes, new_edges

    """Function to compute the acceptable score of the cluster.
    
    Parameters:
        representative_edge_value: (float) The representative value of the good edges in the graph."""

    def compute_acceptable_score(self, representative_edge_value: float) -> None:
        if self.maximum_edges != 0:
            self.acceptable_score = (
                        (self.number_edges * representative_edge_value) * (self.number_edges / self.maximum_edges))
        else:
            self.acceptable_score = 0

    """Function to compute the acceptable ratio of the cluster.
    
    Parameters:
        representative_edge_value: (float) The representative value of the good edges in the graph."""

    def compute_acceptable_ratio(self, representative_edge_value: float) -> None:
        if self.acceptable_score is None:
            self.compute_acceptable_score(representative_edge_value)
        if self.acceptable_score == 0:
            self.acceptable_ratio = 0
        else:
            self.acceptable_ratio = self.score / self.acceptable_score

    """Function to check and remove the isolated nodes from the cluster.
    
    Parameters:
        representative_edge_value: (float) The representative value of the good edges in the graph."""

    def remove_isolated_nodes(self, representative_edge_value: float) -> None:
        new_nodes, new_edges = self._check_isolated_nodes(self.nodes, self.edges)
        self.nodes = new_nodes
        self.edges = new_edges
        self.number_nodes = len(new_nodes)
        self.number_edges = len(new_edges)
        self.score = self._compute_score()
        self.maximum_edges = (self.number_nodes * (self.number_nodes - 1)) / 2
        self.compute_acceptable_score(representative_edge_value)
