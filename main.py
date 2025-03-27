"""Main module for the application.

@Author: Nina Singlan."""
import os
import random
from statistics import mean, stdev
from time import time

from utils.metrics import all_metrics
from utils.save import save_result, save_named_results, compute_and_save_statistics
from utils.utils import parse_args, extract_zip, delete_temporary_directory, read_npz

if __name__ == "__main__":
    args = parse_args()
    global_running_time = time()

    # Check the arguments
    if args.clustering == 'None':
        args.clustering = None

    if args.k == 'None':
        args.k = None

    if not os.path.exists(args.data):
        raise ValueError('The data file does not exist.')

    if not args.data.endswith('.npz') and not args.data.endswith('.zip'):
        raise ValueError('The data file is not correct. It should be either a .npz or a .zip file.')

    if args.no_filter and args.clustering is None:
        raise ValueError('The clustering algorithm should be provided if the filter is not applied.')

    if args.clustering not in ['similarity', 'louvain', None]:
        raise ValueError('The clustering algorithm is not correct. It should be either "similarity" or "louvain".')

    if args.priority not in ['best', 'worst']:
        raise ValueError('The priority is not correct. It should be either "best" or "worst".')

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    use_filter = not args.no_filter

    # Read the data
    if args.verbose:
        print("Starting the experiment...")

    temp_dir = None
    if args.data.endswith('.npz'):
        graphs_paths = [args.data]
    else:
        experiment_name = args.data.split('/')[-1].split('.')[0] + str(random.randint(0, 1000))
        temp_dir, graphs_paths = extract_zip(args.data, experiment_name)

    statistics = {'meanPredicted': [], 'stdevPredicted': [], 'minPredicted': [],
                  'maxPredicted': [], 'time': [], 'meanClusterSize': [], 'stdevClusterSize': [],
                  'minClusterSize': [], 'maxClusterSize': [], 'clusterNumber': [], 'meanClusterScore': [],
                  'partitionScore': []}
    if args.ground_truth:
        statistics = {'F1': [], 'F1Binary': [], 'NMI': [], 'predicted': [], 'meanReal': [], 'stdevReal': [],
                      'minReal': [], 'maxReal': [], 'meanPredicted': [], 'stdevPredicted': [], 'minPredicted': [],
                      'maxPredicted': [], 'time': [], 'meanClusterSize': [], 'stdevClusterSize': [],
                      'minClusterSize': [], 'maxClusterSize': [], 'clusterNumber': [], 'meanClusterScore': [],
                      'partitionScore': []}

        if args.k is not None:
            statistics['F1kScore'] = []
            statistics['F1kBinaryScore'] = []
            statistics['NMIkScore'] = []

    # Start processing the data
    if args.verbose:
        print("Starting to process the data...")

    for graph_path in graphs_paths:
        processing_time = time()

        if args.verbose:
            print(f"Processing the graph {graph_path}...")
            print("Reading the graph...")

        if temp_dir is not None:
            path_graph = temp_dir + '/' + graph_path
        else:
            path_graph = graph_path

        csv_file = None
        if len(graphs_paths) == 1:
            results_file = args.results + '/' + args.output + '.txt'
            if args.name:
                csv_file = args.results + '/' + args.output + '.csv'
        else:
            graph_number = graph_path.split('.')[0]
            results_file = args.results + '/' + args.output + '_graph-' + graph_number + '.txt'
            if args.name:
                csv_file = args.results + '/' + args.output + '_graph-' + graph_number + '.csv'

        graph = read_npz(path_graph, args.ground_truth, args.name, args.threshold)

        computing_time = time()

        # Compute the communities
        if args.verbose:
            print("Starting the clustering algorithm...")

        graph.compute_communities(minimum_nodes=args.min_nodes, use_filter=use_filter,
                                  clustering_algorithm=args.clustering, priority=args.priority)

        if args.community_selection:
            if args.verbose:
                print("Selecting the best community...")
            graph.select_best_communities()

        if args.verbose:
            print("The clustering algorithm is done.")
            computing_time = time() - computing_time
            print(f"The clustering algorithm has been executed in {computing_time // 3600} hours, "
                  f"{(computing_time % 3600) // 60} minutes, and {computing_time % 60} seconds.")

        statistics['time'].append(computing_time)

        communities = graph.communities

        metrics = None
        if args.ground_truth:
            metrics = all_metrics(graph, args.k)

        final_processing_time = time() - processing_time

        cluster_sizes = [len(community) for community in communities]
        if len(cluster_sizes) > 0:
            statistics['meanClusterSize'].append(mean(cluster_sizes))
            statistics['meanClusterScore'].append(mean([community.score for community in communities]))
            if len(cluster_sizes) > 1:
                statistics['stdevClusterSize'].append(stdev(cluster_sizes))
            else:
                statistics['stdevClusterSize'].append(0)
            statistics['minClusterSize'].append(min(cluster_sizes))
            statistics['maxClusterSize'].append(max(cluster_sizes))
        else:
            statistics['meanClusterSize'].append(0)
            statistics['stdevClusterSize'].append(0)
            statistics['minClusterSize'].append(0)
            statistics['maxClusterSize'].append(0)
            statistics['meanClusterScore'].append(0)
        statistics['clusterNumber'].append(len(communities))
        statistics['partitionScore'].append(graph.graph_score)

        mean_F1 = None
        mean_F1k_Score = None
        if args.ground_truth:
            values = [value[0] for value in metrics['F1 Scores'].values()]
            if len(values) > 0:
                mean_F1 = mean(values)
            else:
                mean_F1 = 0
            statistics['F1'].append(mean_F1)
            statistics['F1Binary'].append(metrics['Binary F1 Score'])
            statistics['NMI'].append(metrics['NMI'])
            statistics['predicted'].append(metrics['Predicted'])
            values = [value['mean'] for value in metrics['Information real p-value'].values()]
            if len(values) > 0:
                mean_p_value_real = mean(values)
            else:
                mean_p_value_real = 0
            statistics['meanReal'].append(mean_p_value_real)
            values = [value['stdev'] for value in metrics['Information real p-value'].values()]
            if len(values) > 0:
                stdev_p_value_real = mean(values)
            else:
                stdev_p_value_real = 0
            statistics['stdevReal'].append(stdev_p_value_real)
            values = [value['min'] for value in metrics['Information real p-value'].values()]
            if len(values) > 0:
                min_p_value_real = mean(values)
            else:
                min_p_value_real = 0
            statistics['minReal'].append(min_p_value_real)
            values = [value['max'] for value in metrics['Information real p-value'].values()]
            if len(values) > 0:
                max_p_value_real = mean(values)
            else:
                max_p_value_real = 0
            statistics['maxReal'].append(max_p_value_real)
            values = [value['mean'] for value in metrics['Information predicted p-value'].values()]
            if len(values) > 0:
                mean_p_value_predicted = mean(values)
            else:
                mean_p_value_predicted = 0
            statistics['meanPredicted'].append(mean_p_value_predicted)
            values = [value['stdev'] for value in metrics['Information predicted p-value'].values()]
            if len(values) > 0:
                stdev_p_value_predicted = mean(values)
            else:
                stdev_p_value_predicted = 0
            statistics['stdevPredicted'].append(stdev_p_value_predicted)
            values = [value['min'] for value in metrics['Information predicted p-value'].values()]
            if len(values) > 0:
                min_p_value_predicted = mean(values)
            else:
                min_p_value_predicted = 0
            statistics['minPredicted'].append(min_p_value_predicted)
            values = [value['max'] for value in metrics['Information predicted p-value'].values()]
            if len(values) > 0:
                max_p_value_predicted = mean(values)
            else:
                max_p_value_predicted = 0
            statistics['maxPredicted'].append(max_p_value_predicted)
            if args.k is not None:
                F1k_Score = [value[0] for value in metrics['F1 Scores at k'].values()]
                if len(F1k_Score) > 0:
                    mean_F1k_Score = mean(F1k_Score)
                else:
                    mean_F1k_Score = 0
                statistics['F1kScore'].append(mean_F1k_Score)
                statistics['F1kBinaryScore'].append(metrics['Binary F1 Score at k'])
                statistics['NMIkScore'].append(metrics['NMI at k'])

        if args.verbose:
            print(f"The graph has been processed in {final_processing_time // 3600} hours, "
                  f"{(final_processing_time % 3600) // 60} minutes, and {final_processing_time % 60} seconds.")
        if args.k is not None:
            save_result(results_file, metrics, mean_F1, final_processing_time, graph,
                        mean_F1k_Score)
        else:
            save_result(results_file, metrics, mean_F1, final_processing_time, graph)

        if args.name:
            save_named_results(csv_file, graph)
            save_named_results(csv_file, graph, by_score=True)

        if args.verbose:
            print("Results are saved.")

    if temp_dir is not None:
        delete_temporary_directory(temp_dir)

    if len(graphs_paths) > 1 and args.ground_truth:
        if args.verbose:
            print("Compute and save statistics...")

        statistics_path = args.results + '/' + args.output + '_statistics.txt'
        compute_and_save_statistics(statistics_path, statistics)

        if args.verbose:
            print("Statistics are saved.")

    global_running_time = time() - global_running_time
    if args.verbose:
        print(f"The program has been executed in {global_running_time // 3600} hours, "
              f"{(global_running_time % 3600) // 60} minutes, and {global_running_time % 60} seconds.")

# if __name__ == "__main__":
#     cProfile.run('main()')
