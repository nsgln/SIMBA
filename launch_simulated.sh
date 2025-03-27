#!/bin/bash

# Launch the simulated datasets
# Usage: ./launch_simulated.sh

echo "Launching simulated datasets"
python main.py -d ./data/One_Cluster_Dataset.zip -k 1 -gt -v
python main.py -d ./data/Two_Clusters_Dataset.zip -k 2 -gt -v
python main.py -d ./data/Three_Clusters_Dataset.zip -k 3 -gt -v
python main.py -d ./data/Ten_Clusters_Dataset.zip -k 10 -gt -v
python main.py -d ./data/Medium_Dataset.zip -k 10 -gt -v
