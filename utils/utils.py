"""This file contains utility functions for the project.

@Author: Nina Singlan.

Available functions:
    - parse_args() -> argparse.Namespace
        Parses the command line arguments.
    - similarity(node1: Node, node2: Node, adjacency: List[List[int]]) -> float | None
        Computes the similarity between two nodes.
    - read_npz(file_path: str, ground_truth: bool = True, with_name: bool = False) -> Graph
        Reads the data from the given npz file.
    - extract_zip(zip_file: str, experiment_name: str = '') -> Tuple[str, List[str]
        Extracts the zip file to the given directory.
    - delete_temporary_directory(path: str) -> None
        Deletes the temporary directory created during the extraction of the zip file.
    - sort_communities_by_score(communities: List[Cluster]) -> List[Cluster]
        Sorts the communities by their score.
    - sort_communities_by_graph_score(communities: List[Cluster], representative_value: float) -> List[Cluster]
        Sorts the communities by their graph score.
"""
import argparse
import os
import shutil
import zipfile
from typing import List, Tuple

import numpy as np
import scipy.sparse

from graph.Cluster import Cluster
from graph.Graph import Graph
from graph.Node import Node


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        (argparse.Namespace) The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Community Detection using Graphs.")

    # Input arguments
    parser.add_argument("-d", "--data", dest='data', type=str, required=True,
                        help="Path to the data file, must be npz file, or zip directory containing npz files.")
    parser.add_argument('-gt', '--ground_truth', dest='ground_truth', required=False, action='store_true',
                        help='Flag to indicate that the data file contains ground truth labels.')
    parser.add_argument("-n", "--name", dest='name', required=False, action='store_true',
                        help="Flag to indicate that the data file contains node names.")

    # Output arguments
    parser.add_argument("-res", "--results", dest='results', type=str, required=False, default="results",
                        help="Path to the directory where the results will be saved.")
    parser.add_argument("-o", "--output", dest='output', type=str, required=False, default="output",
                        help="Name of the output file (NO EXTENSION).")

    # Algorithm arguments
    parser.add_argument("-min", "--min", dest='min_nodes', type=int, required=False, default=5,
                        help="The minimum number of nodes in a cluster.")
    parser.add_argument("-no_filter", "--no_filter", dest='no_filter', required=False, action='store_true',
                        help="Flag to indicate that the filter should not be applied.")
    parser.add_argument("-c", "--clustering", dest='clustering', type=str, required=False,
                        default="similarity", choices=["similarity", "louvain", 'None'],
                        help="The clustering algorithm to use.")
    parser.add_argument("-p", "--priority", dest='priority', type=str, required=False, default="worst",
                        choices=["best", "worst"],
                        help="The priority to use in the clustering algorithm (only for similarity-based).")
    parser.add_argument("-cs", "--community_selection", dest='community_selection', required=False,
                        action='store_true', help="Flag to indicate that the community selection should be applied.")
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, required=False, default=0.05,
                        help="The threshold to use on the p-value during the filtering phase. Default is 0.05.")

    # Statistics arguments
    parser.add_argument("-k", "--k", dest='k', type=int, required=False, default=None,
                        help="The number of clusters to find (ONLY USED IN METRIC CALCULATION).")

    # Verbosity arguments
    parser.add_argument("-v", "--verbose", dest='verbose', required=False, action='store_true',
                        help="Flag to indicate that the output should be verbose.")

    return parser.parse_args()


def similarity(node1: Node, node2: Node, adjacency: List[List[int]]) -> float | None:
    """Computes the similarity between two nodes.

    Parameters:
        node1: (Node) The first node.
        node2: (Node) The second node.
        adjacency: (List[List[int]]) The adjacency matrix of the graph.

    Returns:
        (float | None) The similarity between the two nodes."""
    if node1 == node2:
        # No similarity between the same node
        return None
    if adjacency[node1.identifier][node2.identifier] == 0:
        # If there is no edge, similarity is not defined
        return None
    if node1.value + node2.value == 0:
        return float('inf')
    return (1 - abs(node1.value - node2.value)) / (node1.value + node2.value)


def read_npz(file_path: str, ground_truth: bool = True, with_name: bool = False, threshold: float = 0.05) -> Graph:
    """Reads the data from the given npz file.

        Parameters:
            file_path: (str) The path to the npz file.
            ground_truth: (bool) Flag to indicate that the data file contains ground truth labels.
                Optional, default is True.
            with_name: (bool) Flag to indicate that the data file contains node names.
                Optional, default is False.
            threshold: (float) The threshold to use in the filtering.
                Optional, default is 0.05.

        Returns:
            (Graph) The graph object created from the data in the npz file."""
    with np.load(open(file_path, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix(
            (loader['adjacency_data'], loader['adjacency_indices'], loader['adjacency_indptr']),
            shape=loader['adjacency_shape'])

        features = scipy.sparse.csr_matrix(
            (loader['feature_data'], loader['feature_indices'],
             loader['feature_indptr']),
            shape=loader['feature_shape'])

        label_indices = None
        labels = None
        if ground_truth:
            label_indices = loader['label_indices']
            labels = loader['labels']

        names = None
        if with_name:
            names = str(loader['name'])

    assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'

    if ground_truth:
        assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'

    names_dict = None
    if with_name:
        names = names.split(', ')
        names[0] = names[0][1:]
        names[-1] = names[-1][:-1]
        names = [name.split(': ') for name in names]

        names = [[int(name[0]), name[1].replace("'", '')] for name in names]

        names_dict = {}
        for name in names:
            names_dict[name[0]] = name[1]

        assert len(names_dict) == adjacency.shape[0], 'Each node must have a name!'

    # Create nodes
    features_matrix = features.todense()
    features_values = [float(np.asarray(features_matrix[i])[0]) for i in range(adjacency.shape[0])
                       if float(np.asarray(features_matrix[i])[0]) != 0]
    features_values = sorted(features_values)
    epsilon = features_values[0] / 2
    nodes = []
    for i in range(adjacency.shape[0]):
        if float(np.asarray(features_matrix[i])[0]) != 0:
            nodes.append(Node(i, float(np.asarray(features_matrix[i])[0])))
        else:
            nodes.append(Node(i, float(np.asarray(features_matrix[i])[0]) + epsilon))

    adjacency = adjacency.todense()
    adjacency = adjacency.tolist()

    edges = {}

    for i in range(len(adjacency)):
        for j in range(i + 1, len(adjacency[i])):
            if adjacency[i][j] != 0:
                # Since we're working only with undirected graphs, we need to add the edge only once.
                if not (nodes[i], nodes[j]) in edges and not (nodes[j], nodes[i]) in edges:
                    # Using 1 - exp(-sim) allow the edges weights to be in range [0, 1]
                    edges[(nodes[i], nodes[j])] = 1 - np.exp(-similarity(nodes[i], nodes[j], adjacency))

    # Create ground truth
    ground_truth_object = None
    ground_truth_trash = None
    if ground_truth:
        ground_truth_object = []
        ground_truth_trash = set()
        number_of_communities = int(max(labels))

        for i in range(number_of_communities + 1):
            community = [int(label_indices[j]) for j in range(len(label_indices)) if int(labels[j]) == i]
            if i == 0:
                ground_truth_trash = list(set([nodes[j] for j in community]))
            else:
                gt_nodes = list(set([nodes[j] for j in community]))
                gt_edges = {edge: val for edge, val in edges.items() if edge[0] in gt_nodes and edge[1] in gt_nodes}
                ground_truth_object.append(Cluster(gt_nodes, gt_edges))

    return Graph(nodes, edges, threshold, ground_truth_object, ground_truth_trash, names_dict)


def extract_zip(zip_file: str, experiment_name: str = '') -> Tuple[str, List[str]]:
    """Extracts the zip file to the given directory.

        Parameters:
            zip_file: (str) The path to the zip file.
            experiment_name: (str) The name of the experiment.
                Optional, default is ''.

        Returns:
            (Tuple[str, List[str]]) The path to the extracted directory and the list of extracted files."""
    path_to_extract = './tmp'
    if experiment_name:
        path_to_extract = f'./tmp-{experiment_name}'

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract the .zip file.
        zip_ref.extractall(path_to_extract)

    graphs_path = os.listdir(path_to_extract)

    return path_to_extract, graphs_path


def delete_temporary_directory(path: str) -> None:
    """Deletes the temporary directory created during the extraction of the zip file.

        Parameters:
            path: (str) The path to the extracted directory."""
    shutil.rmtree(path)


def sort_communities_by_score(communities: List[Cluster]) -> List[Cluster]:
    """Sorts the communities by their score.

        Parameters:
            communities: (List[Cluster]) The list of communities to sort.

        Returns:
            (List[Cluster]) The sorted list of communities."""
    return sorted(communities, key=lambda x: x.score, reverse=True)


def sort_communities_by_graph_score(communities: List[Cluster], representative_value: float) -> List[Cluster]:
    """Sorts the communities by their graph score.

        Parameters:
            communities: (List[Cluster]) The list of communities to sort.
            representative_value: (float) The representative value of the good edges.

        Returns:
            (List[Cluster]) The sorted list of communities."""
    sorted_communities = []

    for community in communities:
        if community.acceptable_score is None:
            community.compute_acceptable_score(representative_value)
        if community.acceptable_ratio is None:
            community.compute_acceptable_ratio(representative_value)
        community_graph_score = community.acceptable_ratio * community.score

        sorted_communities.append((community, community_graph_score))

    sorted_communities = sorted(sorted_communities, key=lambda x: x[1], reverse=True)

    return [community[0] for community in sorted_communities]
