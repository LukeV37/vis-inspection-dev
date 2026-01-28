import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def eval_model(in_data, in_model, out_dir, model):
    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # Load the preprocessed data
    dataset = np.load(in_data)

    # Split the dataset into train/val/test
    # Determine train/test split base on number of samples
    num_samples=len(dataset)
    test_split=int(0.8*num_samples)
    x_test  = dataset[test_split:]

    # load the trained weights
    model(x_test[0:1])  # build
    model.load_weights(in_model)

    # run the predictions
    pred_image = model.predict(x_test)

    # Save each of the predictions as a JPG
    print("Saving images...")
    for i in tqdm(range(len(pred_image))):
        img = pred_image[i]
        out_path = os.path.join(out_dir, f"pred_{i:04d}.png")
        plt.imsave(out_path, img)
    print(f"Saved {len(pred_image)} images to '{out_dir}'")

if __name__ == "__main__":
    from model import ConvAutoencoder
    model = ConvAutoencoder(embed_dim=64)
    in_data = "../output/dataset.npy"
    in_model = "../output/my_model.weights.h5"
    out_dir = "../output/predictions"
    eval_model(in_data, in_model, out_dir, model)
