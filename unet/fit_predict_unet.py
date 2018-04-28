'''
We use the same methods which was created by our team member Vyom Shrivastava during project 4 with Team Hastings.
'''

import numpy as np
from unet_model import unet
import argparse
from preprocessing_methods import normalize_array
#Importing keras function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

def fit(training_images,mask_images,save_path):
    '''
    This method trains the imaging model via the unet
    @param training_images - input path to training dataset
    @param mask_images - input path to mask files
    @param save_path - path to save the trained resultant data
    '''
    model = unet()
    #Fitting and saving model
    model.fit(training_images, mask_images, batch_size=4, epochs=10, verbose=1, shuffle=True)
    model.save(save_path+"/model.h5")

def predict(testing_images,model_path,save_result_path):
    #loading model and predicting mask
    '''
    Takes in the images to be tested and predicts and saves the result
    '''
    model=unet()
    model.load_weights(model_path)

    prediction = model.predict(testing_images, batch_size=4,verbose=1)
    np.save(save_result_path+"/prediction.npy", prediction)

def fit_predict(mode,image_path,mask_path,model_path,save_path):
    '''
    Calls in the fit and predict method according to the mode which we specified
    '''
    if(mode=="fit"):
        images,mask = normalize_array(image_path,mask_path,mode)
        fit(images,mask,save_path)

    if(mode=="predict"):
        images = normalize_array(image_path,"None",mode)
        predict(images,model_path,save_path)
