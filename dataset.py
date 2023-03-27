import os
import pandas as pd
import pydicom
from torch import from_numpy
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


dict_values = {"foie":63, "rein_droit":126, "rein_gauche":189, "rate":256}

def processMask2Classes(maskArray, dict_Values):
    val = dict_Values["foie"]
    maskArray = maskArray.astype("int")
    maskArray[maskArray !=int(val)]=0
    maskArray[maskArray !=0]=1
    return maskArray

def processMaskMultiClasses(maskArray, dict_Values):
    mask = []
    for key in dict_Values:
        val = dict_Values[key]
        maskSlice = maskArray.astype("int")
        maskSlice[maskSlice !=int(val)]=0
        maskSlice[maskSlice !=0]=1
        mask.append(maskSlice)
    return mask

def getDicom2D(MRPath):
    im = pydicom.dcmread(MRPath).pixel_array.astype("int16")
    return im

def getMask2D(MRPath, mode = "bi"):
    im = np.asarray(Image.open(MRPath)).astype("int16")
    if mode == "multi":
        return processMaskMultiClasses(im,dict_values)
    return processMask2Classes(im, dict_values)


class Dataset(Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, df, transform=None, train = True):
        self.df = df
        self.transform = transform
        self.train = train

    def __len__(self):
        'denotes the total number of samples'
        return(len(self.df.index))

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        path_mr = os.path.join(self.df["root"].iloc[index], self.df["localImPath"].iloc[index])
        X = getDicom2D(path_mr)
        if self.transform:
            X = self.transform(X)

        if self.train:
            path_mask = os.path.join(self.df["root"].iloc[index], self.df["localMaskPath"].iloc[index])
            y = getMask2D(path_mask, mode = "bi")
            if self.transform:
              y = self.transform(y)
            return X,y

        
        return X
    
if __name__ == "__main__":
    df= pd.read_csv(r"C:\Users\nampo\Downloads\Data\train.csv")
    path_mr = os.path.join(df["root"].iloc[0], df["localImPath"].iloc[0])
    path_mask = os.path.join(df["root"].iloc[0], df["localMaskPath"].iloc[0])
    ar1 = getDicom2D(path_mr)
    ar2 = getMask2D(path_mask)
    ar2 = from_numpy(ar2).float()
    print(ar2.shape)