# A new Similarity Based Adapted Louvain Algorithm (SIMBA) for active module identification in p-value attributed biological networks
This repository contains the code needed to execute the SIMBA algorithm and reproduice the results presented in *A new Similarity Based Adapted Louvain Algorithm (SIMBA) for active module identification in p-value attributed biological networks*.

## Table of contents
* [Installation](#installation)
* [Reproduce paper results](#reproduce-paper-results)
* [Usage on your own data](#usage-on-your-own-data)
* [Repository structure](#repository-structure)
* [Data](#data)
* [Requirements](#requirements)
* [Cite](#cite)
* [Contact](#contact)

## Installation
To use this algorithm, it is needed to have Python 3.11 installed on your machine. Moreover, it is needed
to install the required packages described in the [Requirements](#requirements) section and listed in the
``environment.yml`` file.

To run the algorithm, it is needed to clone this repository, install the required packages and run the code
using the commands described in the [Reproduce paper results](#reproduce-paper-results) and the 
[Usage on your own data](#usage-on-your-own-data) sections.

## Reproduce paper results
To reproduce the results of the paper on 'Fully simulated datasets you can use the following command:
```bash
./launch_simulated.sh
```

To reproduce the results of the paper on 'Rewire datasets' you can use the following command:
```bash
./launch_rewire.sh
```

To reproduce the results of the paper on 'Value simulated dataset' you can use the following command:
```bash
./launch_value.sh
```

To reproduce the results of the paper on 'Real dataset' you can use the following command:
```bash
./launch_real.sh
```

## Usage on your own data
To run the code on your own data, you can use the following command:
```bash
python main.py -d ./data/your_data
```

Available options are:
- `-d` or `--data`: **Required** the path to the data file.
- `-gt` or `--ground_truth`: **Optional** flag to indicate that graph contains ground truth.
- `-n` or `--name`: **Optional** flag to indicate that graph contains node names.
- `-res` or `--results`: **Optional** the path to the results directory. Default is `./results/`.
- `-o` or `--output`: **Optional** the path to the output file. Default is `./results/output.txt`.
- `-min` or `--min`: **Optional** the minimum size of the communities. Default is 5.
- `-no_filter` pr `--no_filter`: **Optional** flag to indicate that the filter should not be applied.
- `-c` or `--clustering`: **Optional** the clustering algorithm to use. Default is `similarity`. Available options are `similarity`, `louvain` ang `None`.
- `-p` or `--priority`: **Optional** the priority to use in the clustering algorithm (only for similarity-based). Default is `worst`. Available options are `worst` and `best`.
- `-cs` or `--community_selection`: **Optional** flag to indicate that the community selection should be applied.
- `-t` or `--threshold`: **Optional** the threshold to use in the community selection. Default is 0.05.
- `-k` or `--k`: **Optional** the number of communities to find (used in metrics calculation). Default is `None`.
- `-v` or `--verbose`: **Optional** flag to indicate that the verbose mode should be activated.

## Repository structure
The repository is structured as follows:
- `data/`: contains the data used in the experiments.
- `clustering/`: contains the code of the similarity-based clustering algorithm.
- `graph/`: contains the code to represent the graph structure and the community detection algorithms.
- `utils/`: contains the code of the utility functions used in the experiments.
- `Real_Graphs_Result/`: contains the results of the real datasets experiments.
- `main.py`: contains the main code to run the experiments.
- `Supplementary_Table_S13_all_results.xlsx`: contains the results of the paper.

## Data
### Available data
The data used in the experiments are available in the `data/` folder. Here is the list of the available data:
- `[One_Cluster_Dataset.zip](data/One_Cluster_Dataset.zip)`: contains 1000 graphs of 1000 nodes with 1 community of 10 nodes to find.
- `[Two_Clusters_Dataset.zip](data/Two_Clusters_Dataset.zip)`: contains 1000 graphs of 1000 nodes with 2 communities of 10 nodes to find.
- `[Three_Clusters_Dataset.zip](data/Three_Clusters_Dataset.zip)`: contains 1000 graphs of 1000 nodes with 3 communities of 10 nodes to find.
- `[Ten_Clusters_Dataset.zip](data/Ten_Clusters_Dataset.zip)`: contains 1000 graphs of 1000 nodes with 10 communities of 10 nodes to find.
- `[Medium_Dataset.zip](data/Medium_Dataset.zip)`: contains 100 graphs of 6344 nodes with 10 communities of 10 nodes to find.
- `[Dataset_Rewire_on_Human_0-99_CutOff.zip](data/Dataset_Rewire_on_Human_0-99_CutOff.zip)`: contains 100 graphs of 6344 nodes with 10 communities of 10 nodes to find.
- `[Dataset_Rewire_on_Human_0-8_CutOff.zip](data/Dataset_Rewire_on_Human_0-8_CutOff.zip)`: contains 100 graphs of 14219 nodes with 50 communities of 15 nodes to find.
- `[Tammaro_2024_brain_STRING_v12.npz](data/Tammaro_2024_brain_STRING_v12.npz)`: contains the graph obtained from article: HDAC1/2 inhibitor therapy improves multiple organ systems in aged mice (Tammaro, 2024).
- `[Tammaro_2024_heart_STRING_v12.npz](data/Tammaro_2024_heart_STRING_v12.npz)`: contains the graph obtained from article: HDAC1/2 inhibitor therapy improves multiple organ systems in aged mice (Tammaro, 2024).
- `[Tammaro_2024_kidney_STRING_v12.npz](data/Tammaro_2024_kidney_STRING_v12.npz)`: contains the graph obtained from article: HDAC1/2 inhibitor therapy improves multiple organ systems in aged mice (Tammaro, 2024).

### Data format
Each graph is stored in a `.npz` file. The file contains the following keys:
- `adjacency_data`: the adjacency sparse matrix of the graph.
- `adjacency_indices`: the indices of the adjacency sparse matrix of the graph.
- `adjacency_indptr`: the indptr of the adjacency sparse matrix of the graph.
- `adjacency_shape`: the shape of the adjacency sparse matrix of the graph.
- `feature_data`: the feature sparse matrix of the graph.
- `feature_indices`: the indices of the feature sparse matrix of the graph.
- `feature_indptr`: the indptr of the feature sparse matrix of the graph.
- `feature_shape`: the shape of the feature sparse matrix of the graph.
- `labels`: **Optional - Use with the `-gt` option** the ground truth of the graph.
- `label_indices`: **Optional - Use with the `-gt` option** the ground truth indices of the graph.
- `name`: **Optional - Use with the `-n` option** the name of the nodes.

## Requirements
The code is written in Python 3.11.

The following packages are required to run the code:
- scipy
- scikit-learn
- numpy
- pyunionfind
- scikit-network

## Cite
If you use this code, please cite the following paper:
```bibtex
TO ADD
```

## Contact
If you have any question, please contact me at [singlan.nina@gmail.com](mailto:singlan.nina@gmail.com)