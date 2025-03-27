#!/bin/bash

# Launch the rewire datasets
# Usage: ./launch_rewire.sh

echo "Launching rewire datasets"
python main.py -d ./data/Dataset_Rewire_on_Human_0-99_CutOff.zip -k 10 -gt -v
python main.py -d ./data/Dataset_Rewire_on_Human_0-8_CutOff.zip -k 50 -gt -v