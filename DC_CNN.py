# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:21:15 2023

@author: mallier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size =(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, kernel_size =(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size =(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 128, kernel_size =(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(128*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)


def fit_CNN(net, trainLoader, criterion, optimizer):
    for epoch in range(10):  
        for i, data in enumerate(trainLoader, 0):
            # Get the inputs
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(epoch, loss)
            
    print('Finished Training')
    return net


def val_CNN(net, valLoader):
    correct, total = 0, 0
    predictions = []
    net.eval()
    for i, data in enumerate(valLoader, 0):
        inputs, labels = data
        outputs = net(inputs)    
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))


        

