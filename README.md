# Convolutional Autoencoder Project

## Overview
This project implements a convolutional autoencoder for image reconstruction and analysis. The model compresses input images into a latent space representation and reconstructs them with high fidelity.

## Setup

### Prerequisites
- Conda installed on your system

### Installation
1. Install conda (if not already installed):
   ```bash
   ./conda/install_conda.sh
   ```

2. Setup the conda environment:
   ```bash
   source setup.sh
   ```

## Dataset Preparation
1. Copy or create a symbolic link to your dataset in the `./datasets/` directory:
   ```bash
   cp -r /path/to/R0_DATA_FLEX_F1 ./datasets/
   # OR
   ln -s /path/to/R0_DATA_FLEX_F1 ./datasets/
   ```

## Running the Code
Execute the main script to run the autoencoder:
```bash
python main.py
```

## Project Structure
- `main.py`: Main execution script
- `train/`: Training-related modules
  - `model.py`: Autoencoder model definitions
  - `training.py`: Training logic
- `preprocess/`: Data preprocessing utilities
- `datasets/`: Dataset storage directory
- `conda/`: Conda environment setup files
