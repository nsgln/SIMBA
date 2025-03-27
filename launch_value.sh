#!/bin/bash

# This script launches the value simulated dataset
# Usage: ./launch_value.sh

echo "Launching simulated value dataset"
python main.py -d ./data/Simulated_Value_Dataset.zip -k 10 -gt -v