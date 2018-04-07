import os
import numpy as np
import cv2


#path to data folder
path = "/home/vyom/UGA/DSP/FinalProject/train"

images = []
masks = []
size=[]

#Reading image, combining all mask to a single mask and storing images in a list
for hashVal in hashes:
    hash_path = path+"/"+hashVal+"/"
    image = cv2.imread(hash_path+"images/"+hashVal+".png")
    size.append(image.shape)
    image = cv2.resize(image,(256,256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)
    temp_mask=[]
    mask_images = os.listdir(hash_path+"/masks/")
    for maskImage in mask_images:
        msk = cv2.imread(hash_path+"/masks/"+maskImage)
        msk = cv2.resize(msk,(256,256))
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk[msk==255]=1
        temp_mask.append(msk)
    mask = np.array(temp_mask)
    mask = mask.sum(axis=0,dtype=np.uint8)
    masks.append(mask)

#Changing list to numpy arrays
images=np.array(images)
masks = np.array(masks)
sizes = np.array(size)

#Adding one more channel to the arrays
images = images[...,np.newaxis]
masks = masks[...,np.newaxis]

#Saving numpy arrays to be used later
np.save("test.npy",images)
np.save("masks.npy",masks)
np.save("img_size.npy",sizes)

