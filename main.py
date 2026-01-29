import os
from src.preprocessing import do_preprocessing
from src.model import ConvAutoencoder
from src.training import do_training
from src.eval import eval_model

# Job Parameters
doPreprocessing=True
doTraining=True
doEval=True

# Paths
job_path="output/"
os.makedirs(job_path, exist_ok=True)
dataset_path="./datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/"
preprocess_data=job_path+"dataset.npy"
model_file=job_path+"my_model.weights.h5"
raw_image_dir=job_path+"original_images/"
pred_image_dir=job_path+"pred_images/"

# Model Parameters
latent_dim=1024
epochs=2

# Initialize model
model = ConvAutoencoder(latent_dim)

# Do Jobs
if doPreprocessing:
    do_preprocessing(dataset_path, preprocess_data, raw_image_dir)
if doTraining:
    do_training(preprocess_data, model_file, model, epochs)
if doEval:
    eval_model(preprocess_data, model_file, pred_image_dir, model)
