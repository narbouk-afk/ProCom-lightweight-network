from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plotSingleImageGT(mriPath, gtPath):
    
    mriArray = pydicom.dcmread(mriPath).pixel_array
    gtArray = np.asarray(Image.open(gtPath))
    plt.figure()
    plt.imshow(mriArray)
    plt.imshow(gtArray, alpha=0.5)
    
def showMRIandGT(mriPath, gtPath):
    mriArray = pydicom.dcmread(mriPath).pixel_array
    gtArray = np.asarray(Image.open(gtPath))
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    
    ax1, ax2, ax3 = axs
    
    ax1.imshow(mriArray, cmap='Greys_r')
    ax2.imshow(gtArray)
    ax3.imshow(mriArray, cmap='Greys_r')
    ax3.imshow(gtArray, alpha=0.5)

if __name__ == "__main__":
    
    trainDF = pd.read_csv(r"C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data\train.csv")
    testRow = trainDF.iloc[18]
    mriPath , gtPath = testRow[["imPath", "gtPath"]]
    showMRIandGT(mriPath, gtPath)