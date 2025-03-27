"""This file contains the function needed to perform similarity clustering.

@Author: Nina Singlan.

Available function:
    - _first_phase(nodes: List[Node], edges: Dict[Tuple[Node, Node], float], priority: str, representative_value: float) -> Tuple[List[Cluster], bool]
        Computes the first phase of the similarity clustering algorithm.
    - _second_phase(clusters: List[Cluster], original_edges: Dict[Tuple[Node, Node], float]) -> Tuple[List[Node],
    List[Node, Node, float]]
        Computes the second phase of the similarity clustering algorithm.
    - _original_projection(clusters_trace: List[List[Cluster]]) -> List[Cluster]
        Computes the original projection of the clusters.
    - _connection(edges: Dict[Tuple[Node, Node], float]) -> bool
        Checks if there is a connection between the nodes of the graph.
    - similarity_clustering(nodes: List[Node], edges: Dict[Tuple[Node, Node], float], representative_value: float, priority: str = 'best') -> List[Cluster]
        Computes the similarity clustering algorithm.
    - get_cluster(clusters: List[Cluster], node: Node) -> Cluster
        Returns the cluster containing the given node.
"""
from typing import List, Tuple, Dict

import numpy as np

from graph.Cluster import Cluster
from graph.Node import Node


def _first_phase(nodes: List[Node], edges: Dict[Tuple[Node, Node], float | Tuple[float, int, int | None]],
                 priority: str, representative_value: float, first_run: bool) -> Tuple[List[Cluster], bool]:
    """Computes the first phase of the similarity clustering algorithm.

    Parameters:
        nodes: (List[Node]) The nodes in the graph.
        edges: (Dict[Tuple[Node, Node], float | Tuple[float, int, int | None]]) The edges in the graph.
        priority : (str) The priority of algorithm.
            Can be 'best' or 'worst'.
        representative_value: (float) The representative value of the good edges in the original graph.
        first_run: (bool) Indicates if it is the first run.

    Returns:
        (Tuple[List[Cluster], bool, float]) The clusters, boolean to indicates an update and the score of the graph."""
    # Initialize the clusters
    if first_run:
        clusters = {Cluster([node], {}, avoid_isolated_nodes=False): 0 for node in nodes}
    else:
        clusters = {Cluster([node], {(node, node): edges.get((node, node))}, avoid_isolated_nodes=False):
                        node.value for node in nodes}

    # Initialize the variables
    original_partition = list(clusters.keys())
    partition = original_partition.copy()
    augmented = True
    partition_score = np.sum(list(clusters.values()))

    while augmented:
        augmented = False
        for node in nodes:
            if priority == 'best':
                original_score = partition_score
            else:
                original_score = float('inf')
            best_community = get_cluster(partition, node)
            node_community = best_community
            community_to_remove = None
            community_to_add = None
            increased = False
            new_scores = {'Node': 0, 'Neighbor': 0}
            for neighbor in node.neighbors:
                # Check if the neighbor is in the studied subgraph
                if neighbor in nodes:
                    neighbor_community = get_cluster(partition, neighbor)
                    if neighbor_community != node_community:
                        # Tru to add the node to the neighbor's community
                        new_community_node = node_community.remove_node(node, avoid_isolated_nodes=False)
                        new_community_neighbor = neighbor_community.add_node(node, edges, avoid_isolated_nodes=False)

                        # Compute the new score
                        new_score = 0
                        for cluster in clusters:
                            if node not in cluster.nodes and neighbor not in cluster.nodes:
                                new_score += clusters[cluster]
                            else:
                                if node in cluster:
                                    cluster = new_community_node
                                    name = 'Node'
                                else:  # if neighbor in cluster:
                                    cluster = new_community_neighbor
                                    name = 'Neighbor'
                                if cluster.number_nodes > 1 and cluster.number_edges > 0:
                                    if cluster.acceptable_score is None:
                                        cluster.compute_acceptable_score(representative_value)
                                    if cluster.acceptable_ratio is None:
                                        cluster.compute_acceptable_ratio(representative_value)
                                    score = cluster.acceptable_ratio * cluster.score
                                else:
                                    score = 0
                                new_score += score
                                new_scores[name] = score

                        # Check if the new score is better
                        if priority == 'best':
                            assertion = new_score > original_score
                        else:
                            assertion = original_score > new_score > partition_score
                        if assertion:
                            best_community = new_community_neighbor
                            community_to_remove = [node_community, neighbor_community]
                            community_to_add = new_community_node
                            original_score = new_score
                            augmented = True
                            increased = True
            if increased:
                partition.remove(community_to_remove[0])
                del clusters[community_to_remove[0]]
                partition.remove(community_to_remove[1])
                del clusters[community_to_remove[1]]
                partition.append(best_community)
                clusters[best_community] = new_scores['Neighbor']
                if community_to_add.number_nodes > 0:
                    partition.append(community_to_add)
                    clusters[community_to_add] = new_scores['Node']
                partition_score = original_score

    if original_partition != partition:
        return partition, True
    return partition, False


def _second_phase(clusters: List[Cluster], original_edges: Dict[Tuple[Node, Node], float],
                  representative_value: float) -> Tuple[
    List[Node], Dict[Tuple[Node, Node], Tuple[float, int, int | None]]]:
    """Computes the second phase of the similarity clustering algorithm.

    Parameters:
        clusters: (List[Cluster]) The clusters of the graph.
        original_edges: (Dict[Tuple[Node, Node], float]) The edges of the graph.
        representative_value: (float) A representative value of good edges in original graph.

    Returns:
        (Tuple[List[Node], Dict[Tuple[Node, Node], Tuple[float, int, int | None]]]) The nodes and edges of the graph."""
    # Initialize the variables
    new_nodes = []
    new_edges = {}
    original_nodes = [node for cluster in clusters for node in cluster.nodes]

    raw_edges = False
    if isinstance(list(original_edges.values())[0], (float, np.floating)):
        raw_edges = True

    # Loop through the clusters
    for i in range(len(clusters)):
        if clusters[i].acceptable_score is None:
            clusters[i].compute_acceptable_score(representative_value)
        if clusters[i].acceptable_ratio is None:
            clusters[i].compute_acceptable_ratio(representative_value)
        new_nodes.append(Node(i, clusters[i].score * clusters[i].acceptable_ratio))

    for i in range(len(clusters)):
        cluster = clusters[i]

        # Intra-community edges
        intracommunity_edges = {}
        for node1 in cluster.nodes:
            for node2 in cluster.nodes:
                if node1 != node2:
                    if (node1, node2) in original_edges and (node2, node1) not in intracommunity_edges:
                        intracommunity_edges[(node1, node2)] = original_edges[(node1, node2)]
                    elif (node2, node1) in original_edges and (node1, node2) not in intracommunity_edges:
                        intracommunity_edges[(node1, node2)] = original_edges[(node2, node1)]

        if raw_edges:
            intracommunity_score = np.sum(list(intracommunity_edges.values()))
        else:
            intracommunity_score = np.sum([val[0] for val in intracommunity_edges.values()])
        new_edges[(new_nodes[i], new_nodes[i])] = (intracommunity_score, cluster.number_edges, cluster.number_nodes)

        # Inter-community edges
        not_in_cluster_node = [node for node in original_nodes if node not in cluster.nodes]
        intercommunity_edges = {}
        for node1 in cluster.nodes:
            for node2 in not_in_cluster_node:
                if (node1, node2) in original_edges and (node2, node1) not in intercommunity_edges:
                    intercommunity_edges[(node1, node2)] = original_edges[(node1, node2)]
                elif (node2, node1) in original_edges and (node1, node2) not in intercommunity_edges:
                    intercommunity_edges[(node1, node2)] = original_edges[(node2, node1)]

        for j in range(i + 1, len(clusters)):
            common_edges = {edge: value for edge, value in intercommunity_edges.items()
                            if edge[1] in clusters[j].nodes or edge[0] in clusters[j].nodes}
            if len(common_edges) > 0:
                if raw_edges:
                    intercommunity_score = np.sum(list(common_edges.values()))
                    new_edges[(new_nodes[i], new_nodes[j])] = (intercommunity_score, len(common_edges), None)
                else:
                    intercommunity_score = np.sum([val[0] for val in common_edges.values()])
                    number_edges = np.sum([val[1] for val in common_edges.values()])
                    new_edges[(new_nodes[i], new_nodes[j])] = (intercommunity_score, number_edges, None)

    for edge, value in new_edges.items():
        if edge[0] != edge[1]:
            if edge[0] not in edge[1].neighbors:
                edge[1].add_neighbor(edge[0])
            if edge[1] not in edge[0].neighbors:
                edge[0].add_neighbor(edge[1])

    return new_nodes, new_edges


def _original_projection(clusters_trace: List[List[Cluster]], original_edges: Dict[Tuple[Node, Node], float],
                         ) -> List[Cluster]:
    """Computes the original projection of the clusters.

    Parameters:
        clusters_trace: (List[List[Cluster]]) The clusters of the graph at each iteration.
        original_edges: (Dict[Tuple[Node, Node], float]) The edges of the graph.

    Returns:
        (List[Cluster]) The original projection of the clusters."""
    if len(clusters_trace) == 1:
        return clusters_trace[0]

    # Initialize the variables
    clusters = clusters_trace[-1]

    # Enter the main loop
    for i in range(len(clusters_trace) - 2, -1, -1):
        new_clusters = []
        index = clusters_trace[i]
        for cluster in clusters:
            if type(cluster) == Cluster:
                nodes = cluster.nodes
            else:
                nodes = cluster
            new_cluster_nodes = []
            for node in nodes:
                new_cluster_nodes.extend(index[node.identifier].nodes)
            new_clusters.append(new_cluster_nodes)
        clusters = new_clusters

    final_clusters = []
    for cluster in clusters:
        cluster_nodes = cluster
        cluster_edges = {}
        for node1 in cluster_nodes:
            for node2 in cluster_nodes:
                if (node1, node2) in original_edges and (node2, node1) not in cluster_edges:
                    cluster_edges[(node1, node2)] = original_edges[(node1, node2)]
                elif (node2, node1) in original_edges and (node1, node2) not in cluster_edges:
                    cluster_edges[(node2, node1)] = original_edges[(node2, node1)]

        final_clusters.append(Cluster(cluster_nodes, cluster_edges))

    return final_clusters


def _connection(edges: Dict[Tuple[Node, Node], Tuple[float, int, int | None]]) -> bool:
    """Checks if there is a connection between the nodes of the graph.

    Parameters:
        edges: (Dict[Tuple[Node, Node], float]) The edges of the graph.

    Returns:
        (bool) True if there is a connection between the nodes of the graph, False otherwise."""
    if len(edges) == 0:
        return False

    scores = []
    for e, v in edges.items():
        if e[0] != e[1]:
            scores.append(v[0])

    if len(scores) == 0:
        return False

    if all(score == 0 for score in scores):
        return False

    return True


def _compute_score_communities(communities: List[Cluster], representative_value: float) -> float:
    """Computes the score of the communities.

    Parameters:
        communities: (List[Cluster]) Communities.
        representative_value: (float) The representative value of the good edges in the original graph.

    Return:
        (float) The score."""
    score = 0
    for community in communities:
        if community.acceptable_score is None:
            community.compute_acceptable_score(representative_value)
        if community.acceptable_ratio is None:
            community.compute_acceptable_ratio(representative_value)
        score += community.acceptable_ratio * community.score
    return score


def similarity_clustering(nodes: List[Node], edges: Dict[Tuple[Node, Node], float], representative_value: float,
                          priority: str = 'best') -> List[Cluster]:
    """Computes the similarity clustering algorithm.

    Parameters:
        nodes: (List[Node]) The nodes in the graph.
        edges: (Dict[Tuple[Node, Node], float]) The edges in the graph.
        representative_value: (float) The representative value of the good edges in the original graph.
        priority : (str) The priority of algorithm.
            Optional. Default is 'best'. Can be 'best' or 'worst'.

    Returns:
        (List[Cluster]) The clusters of the graph.

    Raises:
        ValueError: If the priority is not 'best' or 'worst'."""
    if priority not in ['best', 'worst']:
        raise ValueError("The priority must be 'best' or 'worst'.")

    clusters_trace = []
    updated = True
    first_run = True
    original_edges = edges

    while updated:
        # Compute the first phase
        communities, updated = _first_phase(nodes, edges, priority, representative_value, first_run)

        if updated:
            clusters_trace.append(communities)
            # Compute the second phase
            new_nodes, new_edges = _second_phase(communities, original_edges, representative_value)

            if not _connection(new_edges):
                break

            first_run = False

            nodes = new_nodes
            edges = new_edges

    if first_run:
        return [Cluster(nodes, edges)]

    return _original_projection(clusters_trace, original_edges)


def get_cluster(clusters: List[Cluster], node: Node) -> Cluster:
    """Returns the cluster containing the given node.

    Parameters:
        clusters: (List[Cluster]) The list of clusters.
        node: (Node) The node to search for.

    Returns:
        (Cluster) The cluster containing the given node.

    Raises:
        ValueError: If the node is not in any cluster."""
    for cluster in clusters:
        if node in cluster:
            return cluster
    raise ValueError("The node is not in any cluster.")
