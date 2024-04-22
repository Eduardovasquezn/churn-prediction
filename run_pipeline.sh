#!/bin/bash

# Execute feature.py
python ./src/pipeline/feature.py

# Execute training.py
python ./src/pipeline/training.py

# Execute inference.py
python ./src/pipeline/inference.py

# Test inference pipeline by executing test.py
python ./src/test_endpoint/test.py