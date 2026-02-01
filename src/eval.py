import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

def eval_model(in_path, in_model, out_dir, model, batch_size):
    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # Load the dataset
    test_file_paths = sorted(glob.glob(in_path+'Test/preprocessed*.npy'))
    test_dataset = create_dataset(test_file_paths, batch_size=batch_size, shuffle=False)

    names = [os.path.basename(x) for x in test_file_paths]
    tag = [name[12:-4] for name in names]

    # load the trained weights
    dummy_input = tf.zeros((1, 1080, 1920, 3))
    _ = model(dummy_input)
    model.load_weights(in_model)

    # run the predictions
    pred_image = model.predict(test_dataset)

    # Save each of the predictions as a JPG
    print("Saving images...")
    for i in tqdm(range(len(pred_image))):
        img = pred_image[i]
        out_path = os.path.join(out_dir, "pred_"+tag[i]+".png")
        plt.imsave(out_path, img)
    print(f"Saved {len(pred_image)} images to '{out_dir}'")

if __name__ == "__main__":
    from model import ConvAutoencoder
    from data_loader import create_dataset
    model = ConvAutoencoder(embed_dim=64, channels=16)
    in_data = "../output_debug/"
    in_model = "../output_debug/model.weights.h5"
    out_dir = "../output_debug/Predictions"
    batch_size = 32
    eval_model(in_data, in_model, out_dir, model, batch_size)
