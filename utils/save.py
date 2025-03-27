"""File containing the functions needed to save the results.

@Author: Nina Singlan.

Available functions:
    - save_result(path_to_file: str, results: (Dict[str, float | Dict[int, Tuple[float, int | None]] |
    Dict[int, Dict[int, int]] | Dict[int, Dict[str, float]]]), mean_f1: float | None, time: float, graph: Graph) -> None
        Saves the results to a file.
    - compute_and_save_statistics(path_to_file: str, statistics: Dict[str, List[float]]) -> None
        Computes and saves the statistics to a file.
    - save_named_results(path_to_file: str, graph: Graph, by_score: bool = False, by_graph_score: bool = False) -> None
        Saves the results to a file with the node names.
"""
import csv
from statistics import mean, stdev
from typing import Dict, Tuple

from graph.Graph import Graph
from utils.utils import sort_communities_by_score


def save_result(path_to_file: str, results: (Dict[str, float | Dict[int, Tuple[float, int | None]] |
                                                       Dict[int, Dict[int, int]] | Dict[int, float] |
                                                       Dict[int, Dict[str, float]]]), mean_f1: float | None,
                time: float, graph: Graph, mean_f1k: float | None = None) -> None:
    """
    Saves the results to a file.

    Parameters:
        path_to_file: (str) Path to the result file.
        results: (Dict[str, float | Dict[int, Tuple[float, int | None]] | Dict[int, Dict[int, int]] |
                  Dict[int, float] | Dict[int, Dict[str, float]]) The dictionary containing the results
                                                                          to save.
        mean_f1: (float | None) The mean value of F1 Score.
        time: (float) The execution time.
        graph: (Graph) The original graph.
        mean_f1k: (float | None) The mean value of F1 Score for k.
            Optional, default is None.
    """
    if graph.ground_truth is not None:
        if results is None or mean_f1 is None:
            raise ValueError("If the ground truth is present, the metrics should be calculated.")

    with open(path_to_file, 'w+') as file:
        file.write('Number of nodes : ' + str(graph.number_nodes) + '\n')
        file.write('Number of edges : ' + str(graph.number_edges) + '\n')
        if graph.ground_truth is not None:
            file.write('Real number of clusters : ' + str(len(graph.ground_truth)) + '\n')
        file.write('-------------------------------------------' + '\n')
        file.write('Number of clusters find : ' + str(len(graph.communities)) + '\n')
        if len(graph.communities) == 0:
            file.write('No cluster found.' + '\n')
        else:
            for k in range(len(graph.communities)):
                file.write("Cluster {} contains {} nodes and {} edges.\n".format(k, len(graph.communities[k]),
                                                                                 graph.communities[k].number_edges))
                file.write("\tNodes : {}\n".format([node.identifier for node in graph.communities[k].nodes]))
                file.write('\tScore : ' + str(graph.communities[k].score) + '\n')
                file.write('\tAcceptable score : ' + str(graph.communities[k].acceptable_score) + '\n')
        file.write('Score of the partition : ' + str(graph.graph_score) + '\n')
        if graph.ground_truth is not None:
            file.write('--------------------' + '\n')
            file.write('NMI : ' + str(results['NMI']) + '\n')
            file.write('F1-score : ' + str(results['F1 Scores']) + '\n')
            file.write('Mean F1-score : ' + str(mean_f1) + '\n')
            file.write('Binary F1 Score : ' + str(results['Binary F1 Score']) + '\n')
            if mean_f1k is not None:
                file.write('Mean F1 Score at k : ' + str(mean_f1k) + '\n')
                file.write('Binary F1 Score at k :' +
                           str(results['Binary F1 Score at k']) + '\n')
                file.write('NMI at k : ' + str(results['NMI at k']) + '\n')
            file.write('Number of ground truth nodes predicted in each cluster : ' + str(results['Well Predicted']) +
                       '\n')
            file.write('Proportion of ground truth nodes predicted in any cluster : ' + str(results['Predicted']) +
                       '%\n')
            file.write('Information real p-value : ' + str(results['Information real p-value']) + '\n')
            file.write('Information predicted p-value : ' + str(results['Information predicted p-value']) + '\n')
        file.write('Execution time : ' + str(time // 3600) + 'h ' + str((time % 3600) // 60) + 'm ' + str(time % 60)
                   + 's\n')


def compute_and_save_statistics(path_to_file: str, statistics: Dict[str, list[float]]) -> None:
    """
    Computes and saves the statistics to a file.

    Parameters:
        path_to_file: (str) Path to the result file.
        statistics: (Dict[str, list[float]]) The dictionary containing the statistics to save."""
    with open(path_to_file, 'w+') as file:
        for key, values in statistics.items():
            if len(values) == 0:
                file.write(key + ' : ' + 'NO VALUE' + '\n')
            elif key not in ['meanReal', 'stdevReal', 'meanPredicted', 'stdevPredicted', 'meanClusterSize',
                             'stdevClusterSize', 'minClusterSize', 'maxClusterSize']:
                file.write(key + ' : ' + str(mean(values)) + ' +/- ' + str(stdev(values)) + '\n')
            elif key == 'minClusterSize':
                file.write(key + ' : ' + str(min(values)) + '\n')
            elif key == 'maxClusterSize':
                file.write(key + ' : ' + str(max(values)) + '\n')
            elif key in ['meanReal', 'meanPredicted', 'meanClusterSize']:
                file.write(key + ' : ' + str(mean(values)) + ' +/- ')
            else:
                file.write(str(mean(values)) + '\n')


def save_named_results(path_to_file: str, graph: Graph, by_score: bool = False, by_graph_score: bool = False
                       ) -> None:
    """Saves the results to a file with the node names.

    Parameters:
        path_to_file: (str) Path to the result file.
        graph: (Graph) The original graph.
        by_score: (bool) If True, the clusters are sorted by score.
        by_graph_score: (bool) If True, the clusters are sorted by graph score."""
    assert path_to_file.endswith('.csv'), 'The file must be a CSV file.'
    assert graph.nodes_names is not None, 'The nodes must have names.'

    if by_score:
        file = path_to_file.replace('.csv', '_by_score.csv')
        clusters = sort_communities_by_score(graph.communities)
    elif by_graph_score:
        file = path_to_file.replace('.csv', '_by_graph_score.csv')
        clusters = sorted(graph.communities, key=lambda x: x.score, reverse=True)
    else:
        clusters = graph.communities
        file = path_to_file

    header = ['Cluster number', 'Nodes']
    with open(file, 'w+', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i, cluster in enumerate(clusters):
            for node in cluster.nodes:
                if node.identifier in graph.nodes_names:
                    writer.writerow([i, graph.nodes_names[node.identifier]])
                else:
                    writer.writerow([i, 'unknown gene'])
