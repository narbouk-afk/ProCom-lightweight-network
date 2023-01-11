import torch
from torch import nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, alpha=0.5, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.alpha*BCE + self.alpha*dice_loss
        
        return Dice_BCE
