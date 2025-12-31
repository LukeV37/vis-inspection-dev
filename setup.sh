#!/bin/bash
# List conda envs, search for existing install, if not found then create, else continue
conda env list | grep -q "^vis-inspec " || conda env create -f conda/environment.yaml
# Activate the environment
conda activate vis-inspec
