import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *

def classification_model():

    ''' Function to return the classification model which classifies whether mask is present or not in the given image '''

    input = Input((96, 96, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.1)(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)
    
    
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=input, outputs= output)


