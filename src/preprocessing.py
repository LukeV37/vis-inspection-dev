import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import itertools

def rotate_image_about_center(image, angle, scale):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    height, width = (image.shape[0], image.shape[1])
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height),borderValue=(0, 0, 0))
    return rotated_image

def translate_image(image, x_translate, y_translate):
    height, width = (image.shape[0], image.shape[1])
    T_matrix = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
    image_translated = cv.warpAffine(image, T_matrix, (width, height),borderValue=(0, 0, 0))
    return image_translated

def clean_raw_images(in_path, out_path, split=(0.8,0.05,0.15)):
    raw_image_list = glob.glob(in_path+"*.jpg")
    image_ID = [i for i in range(len(raw_image_list))]

    os.makedirs(out_path+"Train", exist_ok=True)
    os.makedirs(out_path+"Val", exist_ok=True)
    os.makedirs(out_path+"Test", exist_ok=True)

    train_split=int(len(raw_image_list)*split[0])
    test_split=int(len(raw_image_list)*(split[0]+split[1]))

    for i in tqdm(range(len(image_ID))):
        ID = image_ID[i]
        image = cv.imread(raw_image_list[i]) # Convert jpg to BGR array (1080,1920,3)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert BGR to RBG array
        clean_image = remove(image)[:,:,0:3] # Remove transparancy layer
        if ID <= train_split:
            save_type="Train"
        elif ID > train_split and ID <= test_split:
            save_type="Val"
        else:
            save_type="Test"

        out_file_npy = os.path.join(out_path, save_type, f"clean_{ID:04d}.npy")
        np.save(out_file_npy, clean_image)

        out_file_png = os.path.join(out_path, save_type, f"clean_{ID:04d}.png")
        plt.imsave(out_file_png, clean_image)

def process_augmentation(params, image, ID, path):
    """Process a single augmentation combination"""
    x, y, r = params

    image_augmented = translate_image(image, x, y)
    image_augmented = rotate_image_about_center(image_augmented, r, 1.0)

    out_file_npy = os.path.join(path, f"preprocessed_{ID}_x{x}_y{y}_angle{r}.npy")
    np.save(out_file_npy, image_augmented)

    out_file_png = os.path.join(path, f"preprocessed_{ID}_x{x}_y{y}_angle{r}.png")
    plt.imsave(out_file_png, image_augmented)
    return True

def augment_dataset(path, max_x, max_y, max_r, n_workers):
    image_list = glob.glob(path+"clean*.npy")
    names = [os.path.basename(x) for x in image_list]
    IDs = [name[6:10] for name in names]

    x_list = [x for x in range(-max_x, max_x+1, 1)]
    y_list = [y for y in range(-max_y, max_y+1, 1)]
    r_list = [r for r in range(-max_r, max_r+1, 1)]

	# Create all parameter combinations
    param_combinations = list(itertools.product(x_list, y_list, r_list))

    for i in tqdm(range(len(image_list))):
        image = np.load(image_list[i])
        ID = IDs[i]

        # Create a partial function with fixed image, ID, and path
        worker_func = partial(process_augmentation, image=image, ID=ID, path=path)

        # Process all combinations in parallel
        with Pool(n_workers) as pool:
            pool.map(worker_func, param_combinations)

if __name__=="__main__":
    data_path = "../datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/"
    out_path= "../output_debug/"
    os.makedirs(out_path, exist_ok=True)
    max_x = 5
    max_y = 5
    max_r = 5
    n_workers = 8
    #clean_raw_images(data_path, out_path)
    for split_type in ["Train/", "Val/", "Test/"]:
        print("Preprocessing ", split_type)
        augment_dataset(out_path+split_type, max_x, max_y, max_r, n_workers)
