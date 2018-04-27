import os
import numpy as np
import cv2
from preprocessing_methods import readImage,resizeImage,lowPassThreshold,medianFiltering,getCombinedMask

def preprocess(mode,path,save_path):

    images = []
    size=[]
    masks = []
    hashes = sorted(os.listdir(path))

    #Reading image, combining all mask to a single mask and storing images in a list
    for hashVal in hashes:
        image =  readImage(path,hashVal)
        size.append(image.shape)
        image = resizeImage(image,256,256)
        image = lowPassThreshold(image,200)
        image = medianFiltering(image,5)
        images.append(image)
        if(mode=="fit"):
            mask = getCombinedMask(path+hashVal+"/masks/")
            masks.append(mask)

    #Changing list to numpy arrays and saving preprocessed images
    images=np.array(images)
    sizes = np.array(size)
    np.save(save_path+"/images.npy",images)
    np.save(save_path+"/img_size.npy",sizes)
    if(mode=="fit"):
        masks = np.array(masks)
        np.save(save_path+"/masks.npy",masks)
