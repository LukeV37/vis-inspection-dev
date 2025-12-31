import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os
from tqdm import tqdm

def load_data_from_path(path):
    # Store all images from path into file_list
    file_list = os.listdir(path)
    # Initialize empty data list
    data_list = []
    # Loop over each file in file list
    for i in tqdm(range(len(file_list))):
        file = file_list[i] # Grab a sinlge image
        image = cv.imread(path+file) # Convert jpg to BGR array (1080,1920,3)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert BGR to RBG array
        data_list.append(image_rgb) # Append the RBG array to data list
    return data_list

### Optional rotation function for dataset augmentation
def rotate_image_about_center(image, angle, scale):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),borderValue=(0, 0, 0))
    return rotated_image

def preprocess_dataset(images, max_angle, angle_step, outpath, save=False):
    # Initialize empy list of preprocessed images
    preprocessed_images = []

    # If performing rotations, generate list of angles
    # By default, max_angle=0 so angle list=[0] and no rotations
    angle_list = [x for x in range(-1*max_angle,max_angle+1,angle_step)]

    # Iterate over images
    for i in tqdm(range(len(images))):
        clean_image = remove(images[i]) # Remove the background using rembg lib
        # If rotating, iterate over angles
        for angle in angle_list:
            # Augment the dataset with a NEW rotated image
            rotated_image = rotate_image_about_center(clean_image, angle, scale=1)
            # Remove the transparency layer and only store RGB values
            rotated_image = rotated_image[:,:,0:3]
            # If save=True; store images as png
            if save:
                plt.imsave(outpath+'output_'+str(i)+'_'+str(angle)+'.png', rotated_image)
            # Append image to preprocessed image list
            preprocessed_images.append(rotated_image)
    # Return an array of preprocessed images as numpy float32
    return np.array(preprocessed_images, dtype='float32')

def do_preprocessing(data_path="../datasets/R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/",
         out_path="./",
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
