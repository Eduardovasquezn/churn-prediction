#!/bin/bash

# Load the .env file
source .env

export PYTHONPATH=$(pwd):$PYTHONPATH

# Execute feature.py
uv run python ./src/pipeline/feature.py

# Execute training.py
uv run python ./src/pipeline/training.py

# Execute inference.py
uv run python ./src/pipeline/inference.py
