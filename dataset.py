import os
import pandas as pd
import pydicom
from torch import from_numpy
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

train = pd.read_csv(r"C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data\train.csv")

def processMask(maskArray, listValues):
    mask = []
    for val in listValues[1:]:
        maskSlice = maskArray.astype("int")
        maskSlice[maskSlice !=int(val)]=0
        maskSlice[maskSlice !=0]=1
        mask.append(maskSlice)
    return np.asarray(mask)

def getDicom3D(MRPath):
    l = []
    for im in os.listdir(MRPath):
        l.append(pydicom.dcmread(os.path.join(MRPath,im)).pixel_array.astype("int16"))
    return np.asarray(l)

def getMask3D(MRPath):
    l = []
    for im in os.listdir(MRPath):
        l.append(np.asarray(Image.open(os.path.join(MRPath,im))).astype("int16"))
    values = np.unique(np.asarray(l))
    for i, im in enumerate(l):
        l[i] = processMask(im,values)
    return np.asarray(l)


class Dataset(Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, df, size=100, transform=False):
        self.df = df
        self.transform = transform
        self.size = size

    def __len__(self):
        'denotes the total number of samples'
        return len(self.df.index)

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        path_mr = os.path.join(self.df["root"].iloc[index], self.df["localImPath"].iloc[index])
        path_mask = os.path.join(self.df["root"].iloc[index], self.df["localMaskPath"].iloc[index])
        X = getDicom3D(path_mr)
        y = getMask3D(path_mask)
        
        X = X[np.newaxis,np.newaxis,:,:,:]  
        X = from_numpy(X).float()
        y = y[np.newaxis,:,:,:]
        y = from_numpy(y).float()

        
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)    
        
        return X, y
    
if __name__ == "__main__":
    df= pd.read_csv(r"C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data\train.csv")
    path_mr = os.path.join(df["root"].iloc[0], df["localImPath"].iloc[0])
    path_mask = os.path.join(df["root"].iloc[0], df["localMaskPath"].iloc[0])
    ar1 = getDicom3D(path_mr)
    ar2 = getMask3D(path_mask)
    ar2 = from_numpy(ar2).float()
    print(ar2.shape)