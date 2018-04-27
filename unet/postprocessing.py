import cv2
import os
import numpy as np
import pandas as pd
from skimage.io import imread

"""
This code for Run length encoding and creating the submission file for submission in kaggle
is referred from the Data Science Bowl 2018 Kaggle Challenge open kernels.
The kernel referred is: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855 
"""
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.2):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def postprocess(image_path,save_path):

    #Reading predicted data and image size information
    sizes_test = np.load(image_path+"/img_size.npy")
    predictions = np.load(image_path+"/prediction.npy")

    grayScale = []
    for image in predictions:
        grayScale.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    grayScale=np.array(grayScale)

    #Setting threhold to the image. Turn pixels below threshold black.
    preds_test_t = (grayScale > 0.5).astype(np.uint8)

    preds_test_upsampled = []
    for i in range(len(predictions)):
        preds_test_upsampled.append(cv2.resize(np.squeeze(preds_test_t[i]),
                                           (sizes_test[i][1], sizes_test[i][0]),
                                           interpolation=cv2.INTER_AREA))
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(save_path+'/submission.csv', index=False)
