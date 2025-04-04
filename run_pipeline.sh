#!/bin/bash

# Load the .env file
source .env

# Execute feature.py
python ./src/pipeline/feature.py

# Execute training.py
python ./src/pipeline/training.py

# Execute inference.py
python ./src/pipeline/inference.py
