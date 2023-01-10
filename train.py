import torch
import torchvision
import torchvision.transforms as transforms
from dataset import Dataset
from torch.optim import Adam
import models 
from loss import DiceBCELoss
import pandas as pd


model = models.UNet3D(in_channels=1, out_channels=3)
criterion = DiceBCELoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1
N_epoch = 20

trainset = pd.read_csv(r"C:\Users\piclt\Desktop\Ecole\4A\ProCom\Data\train.csv")

trainloader = Dataset(trainset)

for epoch in range(N_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 1):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = running_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


print('Finished Training')