"""This file contains the class needed to represent a graph.

@Author: Nina Singlan."""
from statistics import mean, quantiles
from typing import List, Dict, Tuple

import scipy
from numpy import exp
from sknetwork.clustering import Louvain
from unionfind import UnionFind

from clustering.similarity_clustering import similarity_clustering
from graph.Cluster import Cluster
from graph.Node import Node


class Graph:
    """This class represents a graph.

    Attributes:
        nodes: (List[Node]) The nodes in the graph.
        edges: (Dict[Tuple[Node, Node], float]) The edges in the graph.
        number_nodes: (int) The number of nodes in the graph.
        number_edges: (int) The number of edges in the graph.
        communities: (List[Cluster] | None) The communities in the graph.
        trash: (List[Node] | None) The nodes that don't belong to any community.
        ground_truth: (List[Cluster] | None) The ground truth communities.
        ground_truth_trash: (List[Node] | None) The nodes that don't belong to any ground truth community.
        graph_score: (float) The score of the graph.
        nodes_names: (Dict[int, str] | None) The names of the nodes.
        representative_edge_value: (float) The representative value of the good edges."""

    """Initializes the graph with the given nodes and edges.

    Parameters:
        nodes: (List[Node]) The nodes in the graph.
        edges: (Dict[Tuple[Node, Node], float]) The edges in the graph.
        threshold: (float) The threshold to consider an edge.
        ground_truth: (List[Cluster] | None) The ground truth communities
            Optional: Default is None
        ground_truth_trash: (List[Node] | None) The nodes that don't belong to any ground truth community
            Optional: Default is None
        nodes_names: (Dict[int, str]) The names of the nodes.
            Optional: Default is None"""

    def __init__(self, nodes: List[Node], edges: Dict[Tuple[Node, Node], float], threshold: float,
                 ground_truth: List[Cluster] = None, ground_truth_trash: List[Node] = None,
                 nodes_names: Dict[int, str] = None):
        self.nodes = nodes
        self.edges = edges
        self._compute_neighborhood()
        self.number_nodes = len(nodes)
        self.number_edges = len(edges)
        self.communities = None
        self.trash = None
        self.ground_truth = ground_truth
        self.ground_truth_trash = ground_truth_trash
        self.graph_score = 0
        self.nodes_names = nodes_names
        self.representative_edge_value = 0
        self.threshold = threshold

    """Represents the graph as a string.

    Returns:
        (str) A string representation of the graph."""

    def __str__(self) -> str:
        return f"Graph with {self.number_nodes} nodes and {self.number_edges} edges."

    """Computes the neighborhood of the nodes in the graph."""

    def _compute_neighborhood(self) -> None:
        for (node1, node2), value in self.edges.items():
            if node2 not in node1.neighbors:
                node1.add_neighbor(node2)
            if node1 not in node2.neighbors:
                node2.add_neighbor(node1)

    """Function to correct the communities of the graph."""

    def _correct_communities(self) -> None:
        # Add all the edges of the graph to the communities
        for community in self.communities:
            new_edges = {}
            for node1 in community.nodes:
                for node2 in community.nodes:
                    if (node1, node2) in self.edges:
                        new_edges[(node1, node2)] = self.edges[(node1, node2)]
            community.edges = new_edges

        # Remove isolated nodes in communities
        for community in self.communities:
            if community.number_nodes > 1:
                community.remove_isolated_nodes(self.representative_edge_value)

        # Split communities if there is subset inside it.
        new_communities = []
        for community in self.communities:
            if community.number_nodes > 1:
                union_find = UnionFind()
                for (node1, node2), _ in community.edges.items():
                    union_find.union(node1.identifier, node2.identifier)
                components = union_find.components()
                for component in components:
                    nodes = [node for node in community.nodes if node.identifier in component]
                    new_edges = {edge: value for edge, value in community.edges.items()
                                 if edge[0] in nodes and edge[1] in nodes}
                    new_communities.append(Cluster(nodes, new_edges))
            else:
                new_communities.append(community)

        self.communities = new_communities

    """Function to compute the score of the graph.

    Returns:
        (float) The score of the graph."""

    def compute_score(self) -> float:
        if self.communities is None or len(self.communities) == 0:
            self.graph_score = 0
        else:
            graph_score = 0
            for community in self.communities:
                if community.acceptable_score is None:
                    community.compute_acceptable_score(self.representative_edge_value)
                if community.acceptable_ratio is None:
                    community.compute_acceptable_ratio(self.representative_edge_value)
                # Acceptable ratio is used to weight the score of each community
                graph_score += community.acceptable_ratio * community.score
            self.graph_score = graph_score
        return self.graph_score

    """Function to get the adjacency matrix of the graph and the index map to retrieve the node in the matrix.

    Returns:
        (List[List[float]]) The adjacency matrix of the graph.
        (Dict[Node, int]) The index map to retrieve the nodes in the matrix."""

    def get_adjacency_matrix(self) -> Tuple[List[List[float]], Dict[Node, int]]:
        adjacency = [[0 for _ in range(self.number_nodes)] for _ in range(self.number_nodes)]
        index_map = {}
        for i, node in enumerate(self.nodes):
            index_map[node] = i
        for (node1, node2), value in self.edges.items():
            adjacency[index_map[node1]][index_map[node2]] = value
            adjacency[index_map[node2]][index_map[node1]] = value
        return adjacency, index_map

    """Function to filter the graph edges to get only the ones that have a good score.

    Returns:
        (Dict[Tuple[Node, Node], float]) The filtered edges."""

    def _filter_edges(self) -> Dict[Tuple[Node, Node], float]:
        best_edges = {edge: value for edge, value in self.edges.items() if
                      value >= (1 - exp(-(1 / (2 * self.threshold))))}
        if len(best_edges) >= 2:
            self.representative_edge_value = \
            quantiles([value for value in best_edges.values()], n=4, method='inclusive')[2]
        else:
            self.representative_edge_value = mean([value for value in best_edges.values()]) if len(
                best_edges) > 0 else 0
        return best_edges

    """Function to create clusters according to given edges.

    Parameters:
        edges: (Dict[Tuple[Node, Node], float]) The edges to create the clusters from.
        minimum_nodes: (int) The minimum number of nodes in a cluster.
            Optional: Default is 0.

    Returns:
        (List[Cluster]) The clusters created from the edges."""

    def _create_clusters(self, edges: Dict[Tuple[Node, Node], float], minimum_nodes: int = 0) -> List[Cluster]:
        union_find = UnionFind()
        for (node1, node2), _ in edges.items():
            union_find.union(node1.identifier, node2.identifier)
        modules = []
        components = union_find.components()
        for component in components:
            if len(component) >= minimum_nodes:
                nodes = [node for node in self.nodes if node.identifier in component]
                new_edges = {edge: value for edge, value in self.edges.items()
                             if edge[0].identifier in component and edge[1].identifier in component}
                modules.append(Cluster(nodes, new_edges))
        return modules

    """Function to assert if a community should be split.

    Parameters:
        community: (Cluster) The community to check.
        minimum_nodes: (int) The minimum number of nodes in a community.
            Optional: Default is 5.

    Returns:
        (bool) True if the community should be split, False otherwise."""

    def _should_split(self, community: Cluster, minimum_nodes: int = 5) -> bool:
        if community.number_nodes <= minimum_nodes * 2:
            # If the community has less than 2 * minimum_nodes nodes, it shouldn't be split
            return False

        if community.number_nodes > 30 * minimum_nodes:
            return True

        # A community should be split if it has a score lower than the acceptable score
        # The acceptable score is the score of a community with the same number of nodes and edges
        # that has only good edges
        if community.acceptable_score is None:
            community.compute_acceptable_score(self.representative_edge_value)

        return community.score < community.acceptable_score

    """Function to compute the communities of the graph.

    Parameters:
        minimum_nodes: (int) The minimum number of nodes in a community.
            Optional: Default is 0.
        use_filter: (bool) True if the edges should be filtered, False otherwise.
            Optional: Default is True.
        clustering_algorithm: (str | None) The clustering algorithm to use in order to split the communities.
            Optional: Default is None.
            Possible values: 'louvain', 'similarity'.
        priority: (str) The priority of the clustering algorithm.
            Optional: Default is 'best'. Possible values: 'best', 'worst'.

    Raises:
        ValueError: If the clustering algorithm is not valid. Or if the priority is not valid.
        RuntimeError: If the filter isn't used and the clustering algorithm is not specified.
    """

    def compute_communities(self, minimum_nodes: int = 0, use_filter: bool = True, clustering_algorithm: str = None,
                            priority: str = 'best') -> None:
        if clustering_algorithm not in [None, 'louvain', 'similarity']:
            raise ValueError("The clustering algorithm is not valid.")

        if priority not in ['best', 'worst']:
            raise ValueError("The priority is not valid.")

        if not use_filter and clustering_algorithm is None:
            raise RuntimeError("The filter isn't used and the clustering algorithm is not specified.")

        # Even if the filter is not used, we filter the edges to compute the representative_edge_value
        best_edges = self._filter_edges()
        if use_filter:
            self.communities = self._create_clusters(best_edges, minimum_nodes)
            initial_communities = self.communities
            initial_score = self.compute_score()
            updated = True
            if clustering_algorithm is not None:
                while updated:  # Continue until the partition score stops increasing
                    updated = False
                    keep = [False for _ in range(len(self.communities))]
                    communities = self.communities
                    self.communities = []
                    for community in communities:
                        keep_index = 0
                        if self._should_split(community, minimum_nodes):
                            if community.acceptable_score is None:
                                community.compute_acceptable_score(self.representative_edge_value)
                            if community.acceptable_ratio is None:
                                community.compute_acceptable_ratio(self.representative_edge_value)
                            community_initial_graph_score = community.acceptable_ratio * community.score
                            if clustering_algorithm == 'louvain':
                                # Compute the adjacency matrix of the community
                                louvain = Louvain()
                                adjacency, index_map = community.get_adjacency_matrix()
                                adjacency = scipy.sparse.csr_matrix(adjacency)
                                # Compute the communities of the adjacency matrix
                                labels = louvain.fit_predict(adjacency)
                                new_communities = []
                                for label in set(labels):
                                    nodes = [node for node in community.nodes if labels[index_map[node]] == label]
                                    if len(nodes) >= minimum_nodes:
                                        edges = {edge: value for edge, value in community.edges.items() if
                                                 edge[0] in nodes and edge[1] in nodes}
                                        new_communities.append(Cluster(nodes, edges))
                                increased = [self._should_split(c, minimum_nodes) if len(c.nodes) >= 2 * minimum_nodes
                                             else True for c in new_communities]
                                keep[keep_index] = True if False in increased else False
                                self.communities.extend(new_communities)
                            elif clustering_algorithm == 'similarity':
                                new_communities = similarity_clustering(community.nodes, community.edges,
                                                                        self.representative_edge_value,
                                                                        priority)
                                kept_nodes = 0
                                new_communities_score = 0
                                for c in new_communities:
                                    if len(c.nodes) >= minimum_nodes:
                                        kept_nodes += len(c.nodes)
                                        if c.acceptable_score is None:
                                            c.compute_acceptable_score(self.representative_edge_value)
                                        if c.acceptable_ratio is None:
                                            c.compute_acceptable_ratio(self.representative_edge_value)
                                        new_communities_score += c.acceptable_ratio * c.score
                                if (kept_nodes >= community.number_nodes / 5
                                        or new_communities_score > community_initial_graph_score):
                                    self.communities.extend([c for c in new_communities
                                                             if c.number_nodes >= minimum_nodes])
                                    if new_communities_score > community_initial_graph_score:
                                        keep[keep_index] = True
                                    else:
                                        for c in new_communities:
                                            if c.acceptable_score is None:
                                                c.compute_acceptable_score(self.representative_edge_value)
                                        increased = [
                                            c.score < c.acceptable_score for c in new_communities]
                                        keep[keep_index] = True if False in increased else False
                                else:
                                    self.communities.append(community)
                                    keep[keep_index] = False
                            keep_index += 1
                        else:
                            self.communities.append(community)
                    new_score = self.compute_score()
                    same_as_initial = True
                    for com in self.communities:
                        if com not in initial_communities:
                            same_as_initial = False
                            break
                    for com in initial_communities:
                        if com not in self.communities:
                            same_as_initial = False
                            break
                    if (any(keep) or new_score > initial_score) and not same_as_initial:
                        self._correct_communities()
                        initial_communities = self.communities
                        initial_score = new_score
                        updated = True
                    if not updated:
                        self.communities = initial_communities
                        break
        else:
            if clustering_algorithm == 'louvain':
                adjacency, index_map = self.get_adjacency_matrix()
                adjacency = scipy.sparse.csr_matrix(adjacency)
                louvain = Louvain()
                labels = louvain.fit_predict(adjacency)
                self.communities = []
                for label in set(labels):
                    nodes = [node for i, node in enumerate(self.nodes) if labels[i] == label]
                    if len(nodes) >= minimum_nodes:
                        edges = {edge: value for edge, value in self.edges.items()
                                 if edge[0] in nodes and edge[1] in nodes}
                        self.communities.append(Cluster(nodes, edges))
            elif clustering_algorithm == 'similarity':
                self.communities = []
                communities = similarity_clustering(self.nodes, self.edges, self.representative_edge_value,
                                                    priority)
                for community in communities:
                    if len(community.nodes) >= minimum_nodes:
                        self.communities.append(community)

        # Check validity of the communities
        self._correct_communities()

        # Remove communities with less than minimum_nodes nodes
        self.communities = [community for community in self.communities if len(community.nodes) >= minimum_nodes]

        all_nodes = set(self.nodes)
        flattened_communities = [node for community in self.communities for node in community.nodes]
        self.trash = list(all_nodes - set(flattened_communities))
        self.compute_score()

    """Function to select the best communities."""

    def select_best_communities(self):
        if self.communities is not None:
            for community in self.communities:
                if community.acceptable_score is None:
                    community.compute_acceptable_score(self.representative_edge_value)
            self.communities = [community for community in self.communities
                                if community.score >= community.acceptable_score]
            self.compute_score()
