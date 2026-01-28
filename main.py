from preprocess.preprocessing import do_preprocessing
from train.model import ConvAutoencoder
from train.training import do_training
from train.eval import eval_model 

dataset_path="./datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/"
preprocess_path="./preprocess/"

do_preprocessing(dataset_path, preprocess_path)
model = ConvAutoencoder(embed_dim=64)
do_training(preprocess_path+"dataset.npy", model)
eval_model("./preprocess/dataset.npy", out_dir="predictions")