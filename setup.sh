#!/bin/bash

# Store the current directory
current_dir=$(pwd)

# Change to the venv directory
cd venv

# Create venv using python module
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Return to the original directory
cd "$current_dir"
