''''
Usage:
  python eval.py --input ../preprocess/dataset.npy --model my_model.keras --out predictions --split 0.8

What it does: 
- Loads a numpy dataset 
- Splits out the final test fraction (default last 20%)
- Loads a Keras model 
- Runs predictions on x_test
- saves the predictions as jpg files to outfile 

Notes on further improvement: 
Model loading may fail with custom objects or compile settings
Hard-coded model path
No input validation 
Output activation and value range assumptions: that is using a sigmoid function 

'''

import os
import numpy as np
from PIL import Image
import tensorflow as tf

def eval_model(in_path, out_dir="predictions"): 

    # Create the output directory 
    os.makedirs(out_dir, exist_ok=True)
    
    
    # Load the preprocessed data  
    dataset = np.load(in_path)
    
    # Split the dataset into train/val/test
    # Determine train/test split base on number of samples
    num_samples=len(dataset)
    test_split=int(0.8*num_samples)
    x_test  = dataset[test_split:]
    
    # load the trained weights 
    model = ConvAutoencoder(embed_dim=64)
    model(x_test[0:1])  # build
    model.load_weights("my_model.weights.h5")

    
    # run the predictions 
    pred_image = model.predict(x_test)  
    
    # Save each of the predictions as a JPG
    for i, img in enumerate(pred_image): 
        
        # Remove the channel dimensions if grayscale  
        if img.ndim == 3 and img.shape[-1] == 1: 
            img = img.squeeze(-1) 
            
        # Convert from [0,1] float -> [0,255] uint8 
        # Assumes output range is [0,1] since we use the sigmoid function this os true 
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8) 
    
        # Save the image 
        out_path = os.path.join(out_dir, f"pred_{i:04d}.jpg")
        Image.fromarray(img).save(out_path)

    print(f"Saved {len(pred_image)} images to '{out_dir}'")


if __name__ == "__main__":
    eval_model("../preprocess/dataset.npy")
    