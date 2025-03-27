# Data
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