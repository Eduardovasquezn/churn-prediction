#!/bin/bash

# Load the .env file
source .env

# Execute feature.py
python3 ./src/pipeline/feature.py

# Execute training.py
python3 ./src/pipeline/training.py

# Execute inference.py
python3 ./src/pipeline/inference.py
