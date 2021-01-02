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


#Block of code to generate the run length encoder for the predicted masks

image_rows = 420
image_cols = 580


def predict(imgs_test, test_ids, classification_model, double_unet):

    """
    Function to load the model weights and generate the run length encoders form the predicted mask/

    Steps: 
    
    1. Preprocessed test images will be passed to the classification model

    2. If the predicted probability is >0.5 that is the classification model is predicted the presence of mask in the image

    3. Once classification model predicts the image is passed to the Double UNet model for prediction of mask pixels.

   

    """

    classification_model.load_weights('/content/drive/MyDrive/Self_case_study/SC2/Classification_model_weights.h5')
    double_unet.load_weights('/content/drive/MyDrive/Self_case_study/SC2/Double_UNET_model_weights.h5')



    prediction= classification_model.predict(imgs_test[i].reshape(1,96,96,3))[0]

    if (prediction >= 0.5):

        img=double_unet.predict(imgs_test[i].reshape( 1, 96,96,3 ))[0,:,:,0] 

        return img
    else:

        return "No BP nerve detected"


   
