import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

def eval_model(model, test_dataset, data_path, pred_path, batch_size):
    # Get tag from file names
    test_file_paths = sorted(glob.glob(data_path+'Test/preprocessed*.npy'))
    names = [os.path.basename(x) for x in test_file_paths]
    tag = [name[12:-4] for name in names]

    # load the trained weights
    dummy_input = tf.zeros((1, 1080, 1920, 3))
    _ = model(dummy_input)
    model.load_weights(pred_path+"model.weights.h5")

    # run the predictions
    pred_image = model.predict(test_dataset)

    # Save each of the predictions as a JPG
    print("Saving images...")
    for i in tqdm(range(len(pred_image))):
        img = pred_image[i]
        out_path = os.path.join(pred_path, "Predictions", "pred_"+tag[i]+".png")
        plt.imsave(out_path, img)
    print(f"Saved {len(pred_image)} images to '{pred_path}/Predictions'")

if __name__ == "__main__":
    from model import ConvAutoencoder
    from data_loader import create_dataset

    # Parameters
    model = ConvAutoencoder(embed_dim=64, channels=16)
    path = "../output_debug/"
    batch_size = 32

    # Create the output directory
    os.makedirs(path+"Predictions/", exist_ok=True)

    # Load the dataset
    test_file_paths = sorted(glob.glob(path+'Test/preprocessed*.npy'))
    test_dataset = create_dataset(test_file_paths, batch_size=batch_size, shuffle=False)

    eval_model(model, test_dataset, path, path, batch_size)
