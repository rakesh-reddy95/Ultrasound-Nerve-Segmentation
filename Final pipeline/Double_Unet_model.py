
import numpy as np 



import numpy as np
from skimage.transform import resize


import pandas as pd

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from tensorflow.keras.utils import plot_model


tf.keras.backend.clear_session()

def vgg_net(inputs):
    '''
    Encoder1 of the architecture which uses
    the pretrained VGG-19 architecture
    '''
    vgg_19_model=VGG19(include_top=False, weights='imagenet', input_tensor=inputs)

    #We need to take 4 layers which need to be passed to the decoder1
    layer_outputs=[]
    layers=[

        "block1_conv2",  #output shape 64
        "block2_conv2",  #output shape 128
        "block3_conv4",  #output shape 256
        "block4_conv4"   #output shape 512 

        ]
    
    for layer in layers:
        layer_outputs.append(vgg_19_model.get_layer(layer).output)  #output of these layers act as input to the decoder architecuture in network1

    return vgg_19_model.get_layer("block5_conv4").output, layer_outputs #final layer of the VGG19 with output shape 512
    
    


#Altrous Spatial pyramid pooling   
#Reference: https://arxiv.org/pdf/1606.00915v2.pdf  
def ASPP(input):
    ''' ASPP is a semantic segmentation module for resampling a given feature layer at multiple rates prior to convolution. 
     This amounts to probing the original image with multiple filters that have complementary effective fields of view, thus 
     capturing objects as well as useful image context at multiple scales. Rather than actually resampling features, 
     the mapping is implemented using multiple parallel atrous convolutional layers with different sampling rates. '''

    shape=input.shape #(#,#,512 shape)


    a= Conv2D(filters=64, kernel_size=3, dilation_rate=6, padding="same")(input)
    a=BatchNormalization()(a)
    a=Activation("relu")(a)

    b= Conv2D(filters=64, kernel_size=3, dilation_rate=12, padding="same")(input)
    b=BatchNormalization()(b)
    b=Activation("relu")(b)

    c= Conv2D(filters=64, kernel_size=3, dilation_rate=18, padding="same")(input)
    c=BatchNormalization()(c)
    c=Activation("relu")(c)

    d= Conv2D(filters=64, kernel_size=3, dilation_rate=24, padding="same")(input)
    d=BatchNormalization()(d)
    d=Activation("relu")(d)

    x= Concatenate()([a,b,c,d])

    x = Conv2D(64, 3, dilation_rate=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x




def decoder1( input, layer_outputs):
    ''' function to take inputs from output of ASPP and the last 4 layer outputs of VGG19 and Upsample to return the output '''

    filters=[256,128,64,32]
    layer_outputs.reverse()

    x=input

    for index, filter in enumerate(filters):
        x=UpSampling2D((2,2), interpolation='bilinear')(x) #Each block in the decoder performs a 2 × 2 bi-linear up-sampling on the input feature
        x= Concatenate()([x,  layer_outputs[index]]) # concatenate the appropriate skip connections feature maps from the encoder to the output feature maps
        # After concatenation, we again perform two 3 × 3 convolution operation, each of which is followed by batch normalization and then by a ReLU activation function

        x= Conv2D (filter, (3,3), padding='same')(x)

 
        x= BatchNormalization()(x)
        x = Activation ('relu')(x)

        x= Conv2D(filter, (3,3), padding='same')(x)
        x= BatchNormalization()(x)
        x = Activation ('relu')(x)

        #After that, we use a squeeze and excitation block. At last, we apply a convolution layer with a sigmoid activation function, which is used to generate the mask for the corresponding modified U-Net.

        #Ref: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
        ch= x.shape[-1] # getting the last channel
        sq_ex= GlobalAveragePooling2D()(x)
        sq_ex = Dense(ch // 16, activation='relu')(sq_ex)
        sq_ex = Dense(ch, activation='sigmoid')(sq_ex)

        x= Multiply()([x, sq_ex])

    return x


def output_block_network(inputs):
    x=Conv2D(filters=1, kernel_size=(1,1), padding="same")(inputs) #filter =1 to generate the grey scale mask initially
    x= Activation("sigmoid")(x)
    return x



def encoder2(inputs):

    filters=[256,128,64,32]  #reverse of decoder1

    layer_outputs=[]

    x=inputs

    for index, filter in enumerate(filters):
        #Each encoder block in the encoder2 performs two 3 × 3 convolution operation, each followed by a batch normalization
        x= Conv2D (filter, (3,3), padding='same')(x)
        x= BatchNormalization()(x)
        x = Activation ('relu')(x)

        layer_outputs.append(x)

        

        x = MaxPool2D((2, 2))(x)

    return x, layer_outputs


def decoder2(inputs, layer_outputs_encoder1, layer_outputs_encoder2):
    #Here we will only revers the output from the encoder2 not the encoder1. Encoder1 is parallely ppassed as input to the layers of Decoder2.

    filters=[256,128,64,32]
    layer_outputs_encoder2.reverse()

    x= inputs

    for index, filter in enumerate(filters):
        x=UpSampling2D((2,2), interpolation='bilinear')(x) #Each block in the decoder performs a 2 × 2 bi-linear up-sampling on the input feature
        x= Concatenate()([x,  layer_outputs_encoder1[index], layer_outputs_encoder2[index] ]) # concatenate the appropriate skip connections feature maps from the encoder to the output feature maps
        # After concatenation, we again perform two 3 × 3 convolution operation, each of which is followed by batch normalization and then by a ReLU activation function

        x= Conv2D (filter, (3,3), padding='same')(x)

 
        x= BatchNormalization()(x)
        x = Activation ('relu')(x)

        x= Conv2D(filter, (3,3), padding='same')(x)
        x= BatchNormalization()(x)
        x = Activation ('relu')(x)

        #After that, we use a squeeze and excitation block. At last, we apply a convolution layer with a sigmoid activation function, which is used to generate the mask for the corresponding modified U-Net.

        #Ref: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
        ch= x.shape[-1] # getting the last channel
        sq_ex= GlobalAveragePooling2D()(x)
        sq_ex = Dense(ch // 16, activation='relu')(sq_ex)
        sq_ex = Dense(ch, activation='sigmoid')(sq_ex)

        x= Multiply()([x, sq_ex])

    return x








def Double_UNET():
    inputs=Input((96,96,3))
    x, layer_outputs_encoder1= vgg_net(inputs)
    x= ASPP(x)
    x= decoder1(x, layer_outputs_encoder1)
    output_network1= output_block_network(x) #Now input shapes match the output of the Network1.  i.e ( img_row, img_cols, 1)


    #NETWORK 2
    #Output from the previous output will be passed by Multiplying with the input of Network1

    x= Multiply()([inputs, output_network1])
    x, layer_outputs_encoder2 = encoder2(x)
    x= ASPP(x)
    x= decoder2(x, layer_outputs_encoder1, layer_outputs_encoder2)
    output_network2= output_block_network(x) #Now input shapes match the output of the Network2.  i.e ( img_row, img_cols, 1)

    # CONCATENATING OUTPUT FROM NETWORK1 AND NETWORK2
    outputs = tf.maximum(output_network1, output_network2)
    #tf.keras.layers.Add()([output_network1, output_network2])  #Now  output will be  i.e ( img_row, img_cols, 2)



    #Double_UNET = Model(inputs=[inputs], outputs=[outputs])

    return Model(inputs, outputs)


if __name__=='__main__':
    Double_UNET()




