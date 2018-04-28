import cv2
import numpy as np
import os

def readImage(path,hashword):
    '''
    reads image from the path provided
    @param path - path where the image is located
    @param hashword - key for the given image
    '''
    image = cv2.imread(path+hashword+"/images/"+hashword+".png")
    return image

def readMask(path,hashword):
    '''
    reads the mask for the image from the path provided
    @param path - path where the masked images exsist
    @param hashword - mask hash for the mask
    '''
    image = cv2.imread(path+hashword)
    return image

def resizeImage(image,sizeX,sizeY):
    '''
    Takes in an image and resizes it
    @param image - image to be resized
    @param sizeX ,sizeY - new width and height of the image
    '''
    resized = cv2.resize(image,(sizeX,sizeY),interpolation=cv2.INTER_AREA)
    return resized

def bilateralFiltering(image):
    '''
    Takes an image and applies bilateral filtering to the image 
    @image - image on which bilateralFiltering is to be applied
    '''
    filtered = cv2.bilateralFilter(image,10, 75, 75, cv2.BORDER_DEFAULT)
    return filtered

def lowPassThreshold(image,thresholdValue):
    '''
    Applies a lowPassThreshold to an image which allows pixels which are less then the given 
    threshold value
    @param image - image to be considered for filtering
    @param thresholdValue - thresholdValue
    '''
    filtered = image
    filtered[filtered>thresholdValue]=0
    return filtered

def highPassThreshold(image,thresholdValue):
      '''
    Applies a lowPassThreshold to an image which allows pixels which are greater then the given 
    threshold value
    @param image - image to be considered for filtering
    @param thresholdValue - thresholdValue

    '''
    filtered = image
    filtered[filtered<thresholdValue]=0
    return filtered

def bandPassFiler(image,highVal,lowVal):
    '''
    This filter considers the images within the given range
    @image - image on which bandpass filtering needs to be applied
    @param highval - upper bound threshold
    @param lowval - lower bound threshold
    '''
    lower_red = np.array([lowVal,lowVal,lowVal])
    upper_red = np.array([highVal,highVal,highVal])
    filteredMask = cv2.inRange(image, lower_red, upper_red)
    filtered = cv2.bitwise_and(image,image, mask= filteredMask)
    return filtered

def medianFiltering(image,kernelSize):
    '''
    Applies medianFiltering to the image
    @param image - current image to be considered for filtering
    @param kernelSize - size for filtering
    '''
    filtered = cv2.medianBlur(image,kernelSize)
    return filtered

def histogramEqualization(image):
    '''
    histogram equalized image
    '''
    img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    return hist_equalization_result

def laplacian(image):
    '''
    applies laplacian filter to images
    @param image - image to be considered for laplacian filtering
    '''
    laplacian = cv2.Laplacian(image,cv2.CV_32F)
    laplacian[laplacian<0]=0
    return laplacian

def normalize(image):
    '''
    normalizes image values
    @param image - image to be normalized 
    '''
    return image/255

def normalize_array(image_path,mask_path,mode):
    '''
    loads images and mask as a numpy array

    Args:
        image_path:the path of the npy array containing images
                   type: String
        mask_path:      the path to npy array containing masks
                   type: String
    Return:
        numpy array of mask and image
    '''
    data_images = np.load(image_path)
    data_images = normalize(data_images.astype('float32'))

    if(mode=="fit"):
        mask_images = np.load(mask_path)
        mask_images = normalize(mask_images.astype('float32'))
        return data_images, mask_images
    else:
        return data_images

def getCombinedMask(mask_path):
    maskFolder = sorted(os.listdir(mask_path))
    mask = []
    for hashword in maskFolder:
        mask.append(readMask(mask_path,hashword))
    mask = np.array(mask)
    mask = mask.sum(axis=0,dtype=np.uint8)
    mask = resizeImage(mask,256,256)
    mask[mask<255]=0
    return mask
