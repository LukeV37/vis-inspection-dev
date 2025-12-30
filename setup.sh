#!/bin/bash

# Store the current directory
current_dir=$(pwd)

# Change to the venv directory
cd venv

# Activate the virtual environment
source bin/activate

# Install requirements
pip install -r requirements.txt

# Return to the original directory
cd "$current_dir"
