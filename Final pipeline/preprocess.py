import glob
import os.path 
import cv2
import numpy as np 
import matplotlib
from matplotlib.pyplot import  imshow, show
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm

import datetime
from PIL import Image



import numpy as np
from skimage.transform import resize


import pandas as pd

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np


def preprocess_images(imgs, img_rows, img_cols):
    ''' Preprocessing for grey scale images and returns a 3 layer image used in Double-Unet''' 
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
    imgs_p=np.repeat(imgs_p[..., np.newaxis], 3, -1)
    return imgs_p


def preprocess_test_data():

    '''Function to preprocess the test_data to pass to the model as input '''

    # Creating test data to pass to the model
    
    test_images_count=len(os.listdir('test'))

    rows=420
    columns=580
    
    test_images=np.ndarray((test_images_count, rows, columns) ,dtype=np.uint8)
    test_ids=[]



    test_us_images=os.listdir("test/")

    i=0

    for image in tqdm(test_us_images):  # Looping through the list of images and preprocess each image and store in array

        img_id = int(image.split('.')[0])
 
        test_image=cv2.imread(os.path.join("test/",image),0)
        test_image = np.array([test_image])
    
        test_images[i] = test_image
        test_ids.append(img_id)
        i=i+1

    test_ids=np.array(test_ids)    
    np.save('test_ids.npy', test_ids)
    np.save('test_data.npy', test_images)


    #test_images=np.load('test_data.npy')
    #test_ids=np.load('test_ids.npy')
    
    imgs_test = preprocess_images(test_images, 96,96)  #Preprocessing the images to the required dimensions and channels

    imgs_test=imgs_test.astype('float32')

    imgs_test -= 98.14468  #This is train_mean constant used here to standardize and to avoid data leakage
    imgs_test /= 52.407845 #This is train_variance constant used here to standardize and to avoid data leakage
    print("Preprocessing Images completed")
    return imgs_test, test_ids

    
        
