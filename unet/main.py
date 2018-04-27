import numpy as np
import argparse
from preprocessing import preprocess
from fit_predict_unet import fit_predict
from postprocessing import postprocess

parser = argparse.ArgumentParser(description='paths')
parser.add_argument('--mode',type=str,help='fit or predict?')
parser.add_argument('--preprocess',type=str,help='True or False')
parser.add_argument('--postprocess',type=str,help='True or False')
parser.add_argument('--data_path',type=str,help='path of raw data folder')
parser.add_argument('--image_path',type=str,help='path of image npy array')
parser.add_argument('--mask_path',type=str,help='path of mask npy array')
parser.add_argument('--model_path',type=str,help='path of saved model')
parser.add_argument('--save_path',type=str,help='path to save model or prediction')
args = parser.parse_args()

if(args.preprocess=="True"):
    print("Preprocessing Images")
    preprocess(args.mode,args.data_path,args.save_path)
    print("Preprocessing Done!")

elif(args.postprocess=="True"):
    print("postprocessing Images")
    postprocess(args.image_path,args.save_path)
    print("Submission File saved as submission.csv at ",args.save_path)

elif(args.mode=="fit"):
    print("Fitting Model")
    fit_predict(args.mode,args.image_path,args.mask_path,"None",args.save_path)
    print("Done!")

elif(args.mode=="predict"):
    print("Predicting Mask")
    fit_predict(args.mode,args.image_path,"None",args.model_path,args.save_path)
    print("Done!")
