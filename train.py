import torch
import torchvision
import torchvision.transforms as transforms
from dataset import Dataset
from torch.optim import Adam
import models as models
import evaluation as evaluation
from loss import DiceBCELoss
import pandas as pd
import os

def saveModel(model, epoch, loss, savePath): 
    path = os.path.join(savePath, f"epoch{epoch}_loss{loss}_model.pth")
    torch.save(model.state_dict(), path) 


def train(model, criterion, optimizer, N_epoch, 
          trainloader, testloader, transform,
          savePath, device):
    
    best_val_loss = 10**10
    train_losses = []
    val_losses = []
    
    for epoch in range(N_epoch):  # loop over the dataset multiple times

        print(f"epoch {epoch} is starting")
        
        running_loss = 0.0
        running_val_loss = 0.0 
        
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            print(inputs.shape)
            outputs = model(inputs)
            s = outputs.shape[-1]
            labels = transforms.Resize(s, interpolation = transforms.InterpolationMode.NEAREST)(labels)
            labels =  labels.to(device)
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            with torch.no_grad(): 
                model.eval() 
                for data in testloader:  
                   inputs, labels = data
                   inputs, labels = inputs.float(), labels.float()
                   labels = transforms.Resize(s, interpolation = transforms.InterpolationMode.NEAREST)(labels)
                   inputs, labels = inputs.to(device), labels.to(device)
                   predicted_outputs = model(inputs)
                   val_loss = criterion(predicted_outputs, labels) 
                   running_val_loss += val_loss.item()  
        # Save the model if the accuracy is the best S
        if running_val_loss < best_val_loss:
            saveModel(model, epoch, running_val_loss, savePath) 
            best_val_loss = running_val_loss
        
        train_losses.append(running_loss/(len(trainset)))
        val_losses.append(running_val_loss/(len(testset)))
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss, 'Validation Loss is: %.4f' %running_val_loss)
    return train_losses, val_losses


if __name__ == "__main__":
    root = r"D:\IMT Atlantique\TAF\ProCom\rendu\ProCom-lightweight-network\data"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    savePath = r"/Model_save"
    # model = DMFNet(c=1, groups=16, norm='bn', num_classes=1)  # To try DMF network
    model = models.UNet2D()
    model.to(device)
    N_epoch = 20
    batch_size = 16 
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.CenterCrop(size=256)])

    traincsv = pd.read_csv(os.path.join(root, "train2D.csv"))
    traincsv["root"] = root
    traincsv["localImPath"] = traincsv["localImPath"].apply(lambda x: x.replace("\\", "/"))
    traincsv["localMaskPath"] = traincsv["localMaskPath"].apply(lambda x: x.replace("\\", "/")) 
    train80 = traincsv.sample(frac = 0.8, random_state=0)
    test20 = traincsv.drop(train80.index)
    trainset = Dataset(train80, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    testset = Dataset(test20, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True)
    

    
    loss_train, loss_val = train(model, criterion, optimizer, N_epoch, 
          trainloader, testloader, transform,
          savePath, device)
    print('Finished Training')
    
    maps = evaluation.eval(model, testloader, transform, device)
    inputs, outputs, labels = evaluation.processOutput(maps, 15)
    evaluation.showEval(inputs, outputs, labels)