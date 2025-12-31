import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
from tqdm import tqdm

def load_data_from_path(path):
    file_list = os.listdir(path)
    data_list = []
    for file in file_list:
        image = cv.imread(path+file)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        data_list.append(image_rgb)
    return data_list

def rotate_image_about_center(image, angle, scale):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),borderValue=(255, 255, 255))
    return rotated_image

def preprocess_dataset(images):
    rotated_images = []

    angle_list = [x for x in range(-30,31,1)]
    for i in tqdm(range(len(images))):
        clean_image = remove(images[i])
        for angle in angle_list:
            rotated_image = rotate_image_about_center(clean_image, angle, 1)
            rotated_images.append(rotated_image)
    return rotated_images

def convert_to_pytorch_dataset(images):
    return dataset

def main(data_path="../datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/"):
    print("Loading Dataset as numpy arrays...")
    images_list = load_data_from_path(data_path)
    print("Preprocessing Dataset by removing background...")
    images_list_preprocessed = preprocess_dataset(images_list)
    print("Converting Dataset to pyTorch Tensors...")
    pytorch_dataset = convert_to_pytorch_dataset(images_list_augmented)
    return

if __name__=="__main__":
    main()
