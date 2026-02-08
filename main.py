import os
import glob
from src.preprocessing import clean_raw_images
from src.preprocessing import augment_dataset
from src.model import ConvAutoencoder
from src.training import do_training
from src.eval import eval_model
from src.data_loader import create_dataset

# Job Parameters
doCleanRawImages=True
doAugmentDataset=True
doTraining=True
doEval=True

# Preprocessing Parameters
max_x_translation = 50
max_y_translation = 50
max_rotation = 20
n_workers=24
split=(0.8,0.05,0.15)

# Training Parameters
latent_dim=512
channels=2048
epochs=20
batch_size=64

# Preprocessing Paths
preprocess_path="preprocessed_datasets_x"+str(max_x_translation)+"_y"+str(max_y_translation)+"_r"+str(max_rotation)+"/"
raw_dataset_path="./datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/"

# Training Paths
training_path="training_latent"+str(latent_dim)+"_channels"+str(channels)+"_epochs"+str(epochs)+"/"
model_file=training_path+"model.weights.h5"

# Initialize model
model = ConvAutoencoder(latent_dim, channels)

# Do Jobs
if doCleanRawImages:
    os.makedirs(preprocess_path, exist_ok=True)
    clean_raw_images(raw_dataset_path, preprocess_path, split)
if doAugmentDataset:
    os.makedirs(preprocess_path, exist_ok=True)
    for split_type in ["Train/", "Val/", "Test/"]:
        print("Preprocessing ", split_type)
        augment_dataset(preprocess_path+split_type, max_x_translation, max_y_translation, max_rotation, n_workers)
if doTraining:
    os.makedirs(training_path, exist_ok=True)
    train_file_paths = sorted(glob.glob(preprocess_path+'Train/preprocessed*.npy'))
    val_file_paths = sorted(glob.glob(preprocess_path+'Val/preprocessed*.npy'))
    train_dataset = create_dataset(train_file_paths, batch_size=batch_size, shuffle=True)
    val_dataset = create_dataset(val_file_paths, batch_size=batch_size, shuffle=False)
    do_training(model, train_dataset, val_dataset, epochs, batch_size, training_path, len(train_file_paths), len(val_file_paths))
if doEval:
    os.makedirs(training_path+"Predictions/", exist_ok=True)
    test_file_paths = sorted(glob.glob(preprocess_path+'Test/preprocessed*.npy'))
    test_dataset = create_dataset(test_file_paths, batch_size=batch_size, shuffle=False)
    eval_model(model, test_dataset, preprocess_path, training_path, batch_size)
