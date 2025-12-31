#!/bin/bash

# Store the current directory
current_dir=$(pwd)

# Change to the venv directory
cd venv

# Check if environment already exists
if [ ! -f ./venv/bin/activate ]; then
    # Create venv using python module
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate

    # Install requirements
    pip install -r requirements.txt

else
    # Activate the virtual environment
    source venv/bin/activate
fi

# Return to the original directory
cd "$current_dir"
