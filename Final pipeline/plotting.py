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



def grays_to_RGB(img):
    ''' turn 2D grayscale image into grayscale RGB '''
    return np.dstack((img, img, img)) 

def image_with_mask(img, mask):
    ''' to add the mask on the image with outer layer in green '''
    img = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img[mask_edges, 0] = 255  
    img[mask_edges, 1] = 0 
    img[mask_edges, 2] = 0
    return img


def predict_output_plot(classification_model,double_unet, imgs_test ):

    """
    Function to visualize the predicted mask over the original image. 
    """
    classification_model.load_weights('/content/drive/MyDrive/Self_case_study/SC2/Classification_model_weights.h5')
    double_unet.load_weights('/content/drive/MyDrive/Self_case_study/SC2/Double_UNET_model_weights.h5')

    predictions=classification_model.predict(imgs_test)

    for i in range(len(predictions)):
        if predictions[i]>=0.5:
            predictions[i]=1
        else:
            predictions[i]=0

    print("Completed predictions processing")


 
    count=0
    for i in range(len(predictions)):

        if predictions[i]==1:

            pred_mask=np.asarray(double_unet.predict(np.asarray(imgs_test[i].reshape( 1, 96,96,3 )))).reshape(96,96,1)
            pred_mask=((pred_mask[:,:,0]*255.).astype(np.uint8))
            img_test=((imgs_test[i][:,:,0]).astype(np.uint8))

            imshow(image_with_mask(img_test, pred_mask))
            show()
            if(count==10):
                break;
            count+=1


def return_for_deployment(classification_model,double_unet, imgs_test ):

    """
    Function to visualize the predicted mask over the original image. 
    """
    classification_model.load_weights('/content/drive/MyDrive/Self_case_study/SC2/Classification_model_weights.h5')
    double_unet.load_weights('/content/drive/MyDrive/Self_case_study/SC2/Double_UNET_model_weights.h5')

    predictions=classification_model.predict(imgs_test)

    for i in range(len(predictions)):
        if predictions[i]>=0.5:
            predictions[i]=1
        else:
            predictions[i]=0


 
    count=0
    for i in range(len(predictions)):

        if predictions[i]==1:

            pred_mask=np.asarray(double_unet.predict(np.asarray(imgs_test[i].reshape( 1, 96,96,3 )))).reshape(96,96,1)
            pred_mask=((pred_mask[:,:,0]*255.).astype(np.uint8))
            img_test=((imgs_test[i][:,:,0]).astype(np.uint8))

            img= image_with_mask(img_test, pred_mask)

            img = img.astype('float32')
            img = (img > 0.5)
  
            img = resize(img, (256, 256), preserve_range=True)

            #img = resize(img, (580, 420), preserve_range=True)

            return img
       
 
        else:

            return "Empty"

           





