import torch
import torchvision
import torchvision.transforms as transforms
from dataset import Dataset
from torch.optim import Adam
import models 
from loss import DiceBCELoss
import pandas as pd
import os

def saveModel(epoch, loss, savePath): 
    path = os.path.join(savePath, f"epoch{epoch}_loss{loss}_model.pth")
    torch.save(model.state_dict(), path) 


def train(model, criterion, optimizer, trainloader, testloader, N_epoch, savePath, device, transform):
    
    best_val_loss = 0.0
    
    for epoch in range(N_epoch):  # loop over the dataset multiple times

        print(f"epoch {epoch} is starting")
        
        running_loss = 0.0
        running_val_loss = 0.0 
        
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            print("out: ", outputs.shape)
            print("lab :", labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
            with torch.no_grad(): 
                model.eval() 
                for data in testloader: 
                   inputs, outputs = data 
                   inputs = inputs.to(device)
                   predicted_outputs = model(inputs) 
                   val_loss = criterion(predicted_outputs, outputs) 
                 
                   running_val_loss += val_loss.item()  
 
        # Save the model if the accuracy is the best 
        if running_val_loss > best_val_loss: 
            saveModel(epoch, running_val_loss, savePath) 
            best_val_loss = running_val_loss 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss, 'Validation Loss is: %.4f' %running_val_loss)



if __name__ == "__main__":
    root = r"C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    savePath = r"C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data\Save_model"
    model = models.UNet3D(in_channels=1, out_channels=4)
    model.to(device)
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    transform = transforms.Compose(
        [transforms.RandomCrop(size=32)])
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1
    N_epoch = 20

    trainset = pd.read_csv(os.path.join(root, "train.csv"))
    testset = pd.read_csv(os.path.join(root, "test.csv"))
    trainloader = Dataset(trainset, transform=transform)
    testloader = Dataset(testset, transform=transform)
    
    train(model, criterion, optimizer, trainloader, testloader, N_epoch, savePath, device, transform)
    print('Finished Training')