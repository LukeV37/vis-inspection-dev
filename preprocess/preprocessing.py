#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import os

#from tqdm import tqdm


# In[2]:


folder_path="R0_DATA_FLEX_F1/R0_Triplet_Data_Flex_F1_F_White_bg/"


# In[3]:


def load_data_from_path(path):
    file_list = os.listdir(path)
    data_list = []
    for file in file_list:
        image = cv.imread(path+file)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        data_list.append(image_rgb)
    return data_list


# In[4]:


def rotate_image_about_center(image, angle, scale):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image


# In[5]:


def remove_background(image):
    image_rmbg = remove(image)
    return image_rmbg


# In[6]:


image_list = load_data_from_path(folder_path)


# In[ ]:


rotated_images = []

angle_list = [x for x in range(-30,31,1)]
for image in image_list:
    for angle in angle_list:
        rotated_image = rotate_image_about_center(image, angle, 1)
        clean_rotated_image = remove(rotated_image)
        rotated_images.append(rotated_image)
        


# In[ ]:


print(len(rotated_images))


# In[ ]:


image = image_list[0]
plt.imshow(image)
plt.show()

image = remove(image)
plt.imshow(image)
plt.show()


# In[ ]:




