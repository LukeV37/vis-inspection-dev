import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
from tqdm import tqdm

def load_data_from_path(path):
    file_list = os.listdir(path)
    data_list = []
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        image = cv.imread(path+file)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        data_list.append(image_rgb)
    return data_list

def rotate_image_about_center(image, angle, scale):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),borderValue=(0, 0, 0))
    return rotated_image

def preprocess_dataset(images, max_angle, angle_step, outpath, save=False):
    preprocessed_images = []

    angle_list = [x for x in range(-1*max_angle,max_angle+1,angle_step)]
    for i in tqdm(range(len(images))):
        clean_image = remove(images[i])
        for angle in angle_list:
            rotated_image = rotate_image_about_center(clean_image, angle, scale=1)
            rotated_image = rotated_image[:,:,0:3]
            if save:
                plt.imsave(outpath+'output_'+str(i)+'_'+str(angle)+'.png', rotated_image)
            preprocessed_images.append(rotated_image)
    return np.array(preprocessed_images)

def do_preprocessing(data_path="./datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/",
         out_path="./preprocess/",
         max_angle=0, angle_step=1):
    print("Preprocessing path: ", data_path)
    print()
    print("Loading Dataset as numpy arrays...")
    images_list = load_data_from_path(data_path)
    print()
    print("Preprocessing Dataset by removing background...")
    images_list_preprocessed = preprocess_dataset(images_list, max_angle, angle_step, out_path, save=False)
    print()
    print("Shape of Dataset: ", images_list_preprocessed.shape)
    print("Saving dataset as numpy file...")
    np.save(out_path+'dataset.npy', images_list_preprocessed)
    return

if __name__=="__main__":
    do_preprocessing()
