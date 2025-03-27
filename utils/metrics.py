"""File containing the metrics used to evaluate the community detection algorithms.

@Author: Nina Singlan.

Available functions:
    - true_positive(real_cluster: Cluster, predicted_cluster: Cluster) -> int
        Calculates the number of true positives.
    - false_positive(real_cluster: Cluster, predicted_cluster: Cluster) -> int
        Calculates the number of false positives.
    - false_negative(real_cluster: Cluster, predicted_cluster: Cluster) -> int
        Calculates the number of false negatives.
    - precision(real_cluster: Cluster, predicted_cluster: Cluster) -> float
        Calculates the precision.
    - recall(real_cluster: Cluster, predicted_cluster: Cluster) -> float
        Calculates the recall.
    - f1_score(real_cluster: Cluster, predicted_cluster: Cluster) -> float
        Calculates the F1 score.
    - f1_score_all_clusters(real_clusters: List[Cluster], predicted_clusters: List[Cluster], correct: bool = True
                            ) -> Dict[int, Tuple[float, int | None]]
        Calculates the F1 score for all clusters.
    - binary_f1_score(real_clusters: List[Cluster], predicted_clusters: List[Cluster], number_of_nodes: int
                     ) -> Dict[int, Tuple[float, int | None]]
        Calculates the binary F1 score for all clusters.
    - normalized_mutual_information(real_clusters: List[Cluster], predicted_clusters: List[Cluster],
                                    real_trash: List[Node], predicted_trash: List[Node], number_of_nodes: int) -> float
        Calculates the normalized mutual information.
    - well_predicted(real_clusters: List[Cluster], predicted_clusters: List[Cluster]) -> Dict[int, Dict[int, int]]
        Calculates the number of well predicted clusters.
    - predicted(well_predicted_nodes: Dict[int, Dict[int, int]], real_clusters: List[Cluster]) -> float:
        Calculates the percentage of ground truth nodes that are in predicted clusters.
    - information_cluster_p_value(clusters: List[Cluster], graph: Graph) -> Dict[int, Dict[str, float]]
        Calculates the information (mean, standard deviation, min and max) of the p-value of each cluster.
    - all_metrics(real_clusters: List[Cluster], predicted_clusters: List[Cluster], real_trash: List[int],
                    predicted_trash: List[int], number_of_nodes: int
                    ) -> Dict[str, float | Dict[int, Tuple[float, int | None]] | Dict[int, Dict[int, int]]
                    | Dict[int, Dict[str, float]]]
        Calculates all metrics.
"""
from statistics import mean, stdev
from typing import Dict, Tuple, List

from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import normalized_mutual_info_score

from graph.Cluster import Cluster
from graph.Graph import Graph
from graph.Node import Node
from utils.utils import sort_communities_by_score


def true_positive(real_cluster: Cluster, predicted_cluster: Cluster) -> int:
    """Calculates the number of true positives.

    Parameters:
        real_cluster (Cluster): The real cluster.
        predicted_cluster (Cluster): The predicted cluster.

    Returns:
        int: The number of true positives.
    """
    real_cluster_set = set(real_cluster.nodes)
    predicted_cluster_set = set(predicted_cluster.nodes)
    return len(real_cluster_set.intersection(predicted_cluster_set))


def false_positive(real_cluster: Cluster, predicted_cluster: Cluster) -> int:
    """Calculates the number of false positives.

    Parameters:
        real_cluster (Cluster): The real cluster.
        predicted_cluster (Cluster): The predicted cluster.

    Returns:
        int: The number of false positives.
    """
    real_cluster_set = set(real_cluster.nodes)
    predicted_cluster_set = set(predicted_cluster.nodes)
    return len(predicted_cluster_set.difference(real_cluster_set))


def false_negative(real_cluster: Cluster, predicted_cluster: Cluster) -> int:
    """Calculates the number of false negatives.

    Parameters:
        real_cluster (Cluster): The real cluster.
        predicted_cluster (Cluster): The predicted cluster.

    Returns:
        int: The number of false negatives.
    """
    real_cluster_set = set(real_cluster.nodes)
    predicted_cluster_set = set(predicted_cluster.nodes)
    return len(real_cluster_set.difference(predicted_cluster_set))


def precision(real_cluster: Cluster, predicted_cluster: Cluster) -> float:
    """Calculates the precision.

    Parameters:
        real_cluster (Cluster): The real cluster.
        predicted_cluster (Cluster): The predicted cluster.

    Returns:
        float: The precision.
    """
    tp = true_positive(real_cluster, predicted_cluster)
    fp = false_positive(real_cluster, predicted_cluster)
    return tp / (tp + fp) if tp + fp != 0 else 0


def recall(real_cluster: Cluster, predicted_cluster: Cluster) -> float:
    """Calculates the recall.

    Parameters:
        real_cluster (Cluster): The real cluster.
        predicted_cluster (Cluster): The predicted cluster.

    Returns:
        float: The recall.
    """
    tp = true_positive(real_cluster, predicted_cluster)
    fn = false_negative(real_cluster, predicted_cluster)
    return tp / (tp + fn) if tp + fn != 0 else 0


def f1_score(real_cluster: Cluster, predicted_cluster: Cluster) -> float:
    """Calculates the F1 score.

    Parameters:
        real_cluster (Cluster): The real cluster.
        predicted_cluster (Cluster): The predicted cluster.

    Returns:
        float: The F1 score.
    """
    p = precision(real_cluster, predicted_cluster)
    r = recall(real_cluster, predicted_cluster)
    return 2 * p * r / (p + r) if p + r != 0 else 0


def f1_score_all_clusters(real_clusters: List[Cluster], predicted_clusters: List[Cluster], correct: bool = True
                          ) -> dict[int, tuple[float, int | None]]:
    """Calculates the F1 score for all clusters.

    Parameters:
        real_clusters (List[Set[int]]): The real clusters.
        predicted_clusters (List[Set[int]]): The predicted clusters.
        correct (bool): Whether the predicted clusters are correct or not.

    Returns:
        Dict[int, Tuple[float, int | None]]: The F1 score for all clusters.
    """

    def conflict(f1s: Dict[int, Tuple[float, int | None]]) -> bool:
        """Checks if there is a conflict in the F1 scores.

        Parameters:
            f1s: (Dict[int, Tuple[float, int | None]]) The F1 scores.

        Returns:
            (bool) True if there is a conflict, False otherwise."""
        conf = False
        for k in range(len(f1s)):
            for z in range(k + 1, len(f1s)):
                key1 = list(f1s.keys())[k]
                key2 = list(f1s.keys())[z]
                if f1s[key1][1] == f1s[key2][1] and f1s[key1][1] is not None:
                    conf = True
                    break
            if conf:
                break
        return conf

    def update(f1s: Dict[int, Tuple[float, int | None]], scores_saved: Dict[int, List[Tuple[float, int]]],
               number_real_cluster: int) -> Dict[int, Tuple[float, int | None]]:
        """Updates the F1 scores.

        Parameters:
            f1s: (Dict[int, Tuple[float, int | None]]) The F1 scores.
            scores_saved: (Dict[int, List[Tuple[float, int]]]) The saved scores.
            number_real_cluster: (int) The number of real clusters.

        Returns:
            (Dict[int, Tuple[float, int | None]]) The updated F1 scores."""
        updated_f1s = {}
        clusters_to_change = []

        already_seen = [False for _ in range(number_real_cluster)]

        for cluster in f1s.keys():
            cluster_ground_truth = f1s[cluster][1]
            if cluster_ground_truth is not None:
                if already_seen[cluster_ground_truth]:
                    other_clusters = []
                    for other_cluster in f1s.keys():
                        if f1s[other_cluster][1] == cluster_ground_truth:
                            other_clusters.append(other_cluster)

                    best = -1
                    best_cluster = None
                    for other_cluster in other_clusters:
                        if f1s[other_cluster][0] > best:
                            best = f1s[other_cluster][0]
                            best_cluster = other_cluster

                    updated_f1s[best_cluster] = (f1s[best_cluster][0], f1s[best_cluster][1])
                    for other_cluster in other_clusters:
                        if other_cluster != best_cluster:
                            clusters_to_change.append(other_cluster)

                else:
                    already_seen[cluster_ground_truth] = True

        for cluster in clusters_to_change:
            actual_ground_truth = f1s[cluster][1]
            actual_past = False
            updated = False

            for r in range(len(scores_saved[cluster])):
                if actual_past:
                    updated_f1s[cluster] = (scores_saved[cluster][r][0], scores_saved[cluster][r][1])
                    updated = True
                    break
                else:
                    if scores_saved[cluster][r][1] == actual_ground_truth:
                        actual_past = True

            if not updated:
                updated_f1s[cluster] = (0, None)

        for cluster in f1s.keys():
            if cluster not in updated_f1s:
                updated_f1s[cluster] = f1s[cluster]

        return updated_f1s

    # WARNING: This function is not optimized and can be slow when number of found clusters is high.
    f1 = {}
    scores = {}
    for i in range(len(predicted_clusters)):
        score = []
        for j in range(len(real_clusters)):
            score.append(f1_score(real_clusters[j], predicted_clusters[i]))
        f1_value = max(score)
        f1_index = score.index(f1_value)
        if f1_value != 0:
            f1[predicted_clusters[i].identifier] = (f1_value, f1_index)
        else:
            f1[predicted_clusters[i].identifier] = (0, None)
        if len(score) == len(set(score)):
            s = [(value, score.index(value)) for value in score]
        else:
            # Some scores are the same, we need to keep the index of the real cluster
            s = []
            for j in range(len(score)):
                s.append((score[j], j))
        scores[predicted_clusters[i].identifier] = sorted(s, reverse=True, key=lambda x: x[0])

    if correct:
        while conflict(f1):
            f1 = update(f1, scores, len(real_clusters))

    # Add a penalty for the clusters that are not found
    if len(f1) < len(real_clusters):
        for i in range(len(f1), len(real_clusters)):
            f1[-i] = (0, None)

    return f1


def binary_f1_score(real_clusters: List[Cluster], predicted_clusters: List[Cluster], number_of_nodes: int
                    ) -> dict[int, tuple[float, int | None]]:
    """Calculates the binary F1 score for all clusters.

    Parameters:
        real_clusters (List[Set[int]]): The real clusters.
        predicted_clusters (List[Set[int]]): The predicted clusters.
        number_of_nodes (int): The number of nodes.

    Returns:
        Dict[int, Tuple[float, int | None]]: The binary F1 score for all clusters.
    """
    real = [0 for _ in range(number_of_nodes)]
    pred = [0 for _ in range(number_of_nodes)]

    for i in range(len(real_clusters)):
        for node in real_clusters[i].nodes:
            real[node.identifier] = 1

    for i in range(len(predicted_clusters)):
        for node in predicted_clusters[i].nodes:
            pred[node.identifier] = 1

    return f1_score_sklearn(real, pred)


def normalized_mutual_information(real_clusters: List[Cluster], predicted_clusters: List[Cluster],
                                  real_trash: List[Node], predicted_trash: List[Node], number_of_nodes: int) -> float:
    """Calculates the normalized mutual information.

    Parameters:
        real_clusters (List[Set[int]]): The real clusters.
        predicted_clusters (List[Set[int]]): The predicted clusters.
        real_trash (List[int]): The real trash nodes.
        predicted_trash (List[int]): The predicted trash nodes.
        number_of_nodes (int): The number of nodes.

    Returns:
        float: The normalized mutual information.
    """
    real = [0 for _ in range(number_of_nodes)]
    pred = [0 for _ in range(number_of_nodes)]

    for i in range(len(real_clusters)):
        for node in real_clusters[i].nodes:
            real[node.identifier] = i + 1

    for i in range(len(predicted_clusters)):
        for node in predicted_clusters[i].nodes:
            pred[node.identifier] = i + 1

    for node in real_trash:
        real[node.identifier] = 0

    for node in predicted_trash:
        pred[node.identifier] = 0

    return normalized_mutual_info_score(real, pred)


def well_predicted(real_clusters: List[Cluster], predicted_clusters: List[Cluster]) -> Dict[int, Dict[int, int]]:
    """Calculates the number of well predicted clusters.

    Parameters:
        real_clusters (List[Set[int]]): The real clusters.
        predicted_clusters (List[Set[int]]): The predicted clusters.

    Returns:
        int: The number of well predicted clusters.
    """
    well_predicted_nodes = {}
    for i in range(len(real_clusters)):
        well_predicted_nodes[i] = {}
        for j in range(len(predicted_clusters)):
            set_real = set(real_clusters[i].nodes)
            set_predicted = set(predicted_clusters[j].nodes)
            well_predicted_nodes[i][j] = len(set_real.intersection(set_predicted))
    return well_predicted_nodes


def predicted(well_predicted_nodes: Dict[int, Dict[int, int]], real_clusters: List[Cluster]) -> float:
    """Calculates the percentage of ground truth nodes that are in predicted clusters.

    Parameters:
        well_predicted_nodes (Dict[int, Dict[int, int]]): The number of well predicted nodes.
        real_clusters (List[Set[int]]): The real clusters.

    Returns:
        Dict[int, int]: The percentage of ground truth nodes that are in predicted clusters.
    """
    number_of_nodes = 0
    total = 0
    for i in range(len(real_clusters)):
        number_of_nodes += len(real_clusters[i])
        for j in range(len(well_predicted_nodes[i])):
            total += well_predicted_nodes[i][j]
    return (total / number_of_nodes) * 100


def information_cluster_p_value(clusters: List[Cluster]) -> Dict[int, Dict[str, float]]:
    """Calculates the information (mean, standard deviation, min and max) of the p-value of each cluster.

    Parameters:
        clusters (List[Cluster]): The clusters.

    Returns:
        Dict[int, Dict[str, float]]: The information of the p-value of each cluster."""
    mean_p_value = {}
    for i in range(len(clusters)):
        if len(clusters[i]) == 0:
            mean_p_value[i] = {'mean': 0, 'stdev': 0, 'min': 0, 'max': 0}
            continue
        else:
            info = {}
            p_value = []
            for node in clusters[i].nodes:
                p_value.append(node.value)
            if len(p_value) > 1:
                info['mean'] = mean(p_value)
                info['stdev'] = stdev(p_value)
                info['min'] = min(p_value)
                info['max'] = max(p_value)
            else:
                info['mean'] = p_value[0]
                info['stdev'] = 0.0
                info['min'] = p_value[0]
                info['max'] = p_value[0]
            mean_p_value[i] = info
    return mean_p_value


def all_metrics(graph: Graph, k: int | None = None) -> (
Dict[str, float | Dict[int, Tuple[float, int | None]] | Dict[int, Dict[int, int]]
          | Dict[int, Dict[str, float]]]):
    """Calculates all metrics.

    Parameters:
        graph (Graph): The graph.
        k (int | None): The number of clusters to find. Optional, default is None.

    Returns:
        (Dict[str, float | Dict[int, Tuple[float, int | None]] | Dict[int, Dict[int, int]]
        | Dict[int, Dict[str, float]]): The metrics."""

    if graph.ground_truth is None:
        raise ValueError("The ground truth is needed to compute the metrics.")

    real_clusters = graph.ground_truth
    predicted_clusters = graph.communities
    real_trash = graph.trash
    predicted_trash = graph.trash
    number_of_nodes = graph.number_nodes
    predict = well_predicted(real_clusters, predicted_clusters)
    results = {'F1 Scores': f1_score_all_clusters(real_clusters, predicted_clusters),
               'Binary F1 Score': binary_f1_score(real_clusters, predicted_clusters, number_of_nodes),
               'NMI': normalized_mutual_information(real_clusters, predicted_clusters, real_trash, predicted_trash,
                                                    number_of_nodes),
               'Well Predicted': predict,
               'Predicted': predicted(predict, real_clusters),
               'Information predicted p-value': information_cluster_p_value(predicted_clusters),
               'Information real p-value': information_cluster_p_value(real_clusters)
               }
    if k is not None:
        communities_sorted_by_score = sort_communities_by_score(graph.communities)[:k]
        results['F1 Scores at k'] = f1_score_all_clusters(real_clusters, communities_sorted_by_score)
        results['Binary F1 Score at k'] = binary_f1_score(real_clusters, communities_sorted_by_score,
                                                          number_of_nodes)
        results['NMI at k'] = normalized_mutual_information(real_clusters, communities_sorted_by_score,
                                                            real_trash, predicted_trash, number_of_nodes)

    return results
