#!/bin/bash

# Store the current directory
current_dir=$(pwd)

# Change to the venv directory
cd venv

# Install requirements
pip install -r requirements.txt

# Return to the original directory
cd "$current_dir"
