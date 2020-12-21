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

def prep(img):
    img = img.astype('float32')
    img = (img > 0.5)
  
    img = resize(img, (image_cols, image_rows), preserve_range=True)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])




def final(imgs_test, test_ids, classification_model, double_unet):

    """
    Function to load the model weights and generate the run length encoders form the predicted mask/

    Steps: 
    
    1. Preprocessed test images will be passed to the classification model

    2. If the predicted probability is >0.5 that is the classification model is predicted the presence of mask in the image

    3. Once classification model predicts the image is passed to the Double UNet model for prediction of mask pixels.

    4. With the function for generating the run length encoders this will generate the file. 

    """

    classification_model.load_weights('Classification_model_weights.h5')
    double_unet.load_weights('Double_UNET_model_weights.h5')
    argsort = np.argsort(test_ids)
    imgs_id_test = test_ids[argsort]
    imgs_test = imgs_test[argsort]
    first_row = 'img,pixels'

    with open('submission_file','w+') as f:
        f.write(first_row+'\n')
        for i in tqdm(range(imgs_test.shape[0])):
            prediction= classification_model.predict(imgs_test[i].reshape(1,96,96,3))[0]

            if (prediction>=0.5):
            
                img=double_unet.predict(imgs_test[i].reshape( 1, 96,96,3 ))[0,:,:,0]            
                img=prep(img)
                rle=run_length_enc(img)
                s=str(imgs_id_test[i])+','+rle 
                f.write(s + '\n')

            else:
           
                s=str(imgs_id_test[i])+','+  ''
                f.write(s + '\n')

    print("Succesfully generated the file")
   