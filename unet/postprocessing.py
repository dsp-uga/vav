import os
import numpy as np
import cv2

#path to npy outputs
path = "/home/vyom/UGA/DSP/FinalProject/"

#Getting all hash names
folder = sorted(os.listdir(path))

#Reading predicted data and image size information
size = np.load(path+"img_size.npy")
image= np.load(path+"prediction.npy")

#Setting threhold to the image. Turn pixels below threshold black.
def threshold(image,t):
    _, new_image = cv2.threshold(image,t,255,cv2.THRESH_BINARY)
    return new_image

for i in range (len(folder)):
    new = threshold(image[i],0.95)
    new=cv2.resize(new,(size[i][1],size[i][0]))
    new[new==255]=1
    cv2.imwrite("./predictionsToShow/"+folder[i].split("\n")[0]+".png",new)
