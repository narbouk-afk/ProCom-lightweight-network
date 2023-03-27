from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.nn.functional import sigmoid
def plotCurves(train, test):
    x = np.arange(1,21,1)
    plt.plot(x,train)
    plt.plot(x,test)
    plt.legend(["train", "test"])
    plt.show()

def eval(model, testloader, transform, device):
    model.eval()
    model.to(device)
    best_val_loss = 10**10
    train_losses = []
    val_losses = []
    maps = []
    for i, data in enumerate(testloader, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.float(), labels.float()
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        maps.append((inputs, outputs, labels))

    return maps
    
def showEval(inputs, outputs, labels):
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    
    ax1, ax2 = axs
    
    ax1.imshow(inputs, cmap='Greys_r')
    ax1.imshow(outputs, alpha=0.5)
    ax2.imshow(inputs, cmap='Greys_r')
    ax2.imshow(labels, alpha=0.5)

def processOutput(maps, n):
    inputs, outputs, labels = maps[n]
    s = outputs.shape[-1]
    print(s)
    inputs = transforms.Resize(s)(inputs)
    labels = transforms.Resize(s, interpolation = transforms.InterpolationMode.NEAREST)(labels)
    inputs = inputs.cpu().detach().numpy()[0,0,:,:]
    labels = labels.cpu().detach().numpy()[0,0,:,:]
    outputs = sigmoid(outputs).cpu().detach().numpy()[0,0,:,:]
    outputs[outputs<0.5]=0
    outputs[outputs>=0.5]=1
    return inputs, outputs, labels
