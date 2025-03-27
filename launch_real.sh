#!/bin/bash

# Launch the real datasets
# Usage: ./launch_real.sh

echo "Launching real datasets"
python main.py -d ./data/Tammaro_2024_brain_STRING_v12.npz -n -v
python main.py -d ./data/Tammaro_2024_heart_STRING_v12.npz -n -v
python main.py -d ./data/Tammaro_2024_kidney_STRING_v12.npz -n -v