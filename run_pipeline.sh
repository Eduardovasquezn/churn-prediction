#!/bin/bash
echo "PYTHONPATH=$PYTHONPATH:$(pwd)/"
# Load the .env file
source .env

# Set python path
echo "PYTHONPATH=$PYTHONPATH:$(pwd)/"

# Execute feature.py
python3 ./src/pipeline/feature.py

# Execute training.py
python3 ./src/pipeline/training.py

# Execute inference.py
python3 ./src/pipeline/inference.py
