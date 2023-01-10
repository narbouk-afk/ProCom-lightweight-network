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

def getDicom3D(MRPath):
    l = []
    for im in os.listdir(MRPath):
        l.append(pydicom.dcmread(os.path.join(MRPath,im)).pixel_array.astype("int16"))
    return np.asarray(l)

def getMask3D(MRPath):
    l = []
    for im in os.listdir(MRPath):
        l.append(np.asarray(Image.open(os.path.join(MRPath,im))).astype("int16"))
    return np.asarray(l)

def ResizeGridSample(X_init, size, is_mask=False):
    shape = X_init.shape
    grid = torch.tensor([shape[0], size, size, 2]).float()
    print(grid)
    if is_mask:
        return F.grid_sample(X_init, grid, mode='nearest')
    
    return F.grid_sample(X_init, grid)
         
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
        y = y[np.newaxis,np.newaxis,:,:,:]
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
    ar1 = from_numpy(ar1).float()
    