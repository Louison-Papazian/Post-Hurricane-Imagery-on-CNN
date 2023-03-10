# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:56:05 2023

@author: mallier

Méthode medium.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


#Fonction permettant d'obtenir l'accuracy des modèles
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


#Création d'une classe de suivis des modèles avec prediction sur les données de validation
#et metrics d'analyse
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


#Model de l'article en pytorch avec comme base la classe de suivi des modèles
class MediumCNN(ImageClassificationBase):
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
    
#rajout dropout   
class ConvNet(ImageClassificationBase):
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
            nn.Dropout(p=0.5),
            nn.Linear(128*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid())

    def forward(self, xb):
        return self.network(xb)

#Evaluation du modèle pour l'entrainement
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = []
    step = 0
    for batch in val_loader:
        outputs.append(model.validation_step(batch))
        step += 1
        if step == 50:
            break
        
    return model.validation_epoch_end(outputs)


#Fonction d'entrainement sur le dataloader train avec epoch
def MediumFit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        step = 0
        for batch in train_loader: 
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step == 100:
                break
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history, model


#modification du CNN de base
class ConvNet_mod(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size =(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, kernel_size =(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size =(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 128, kernel_size =(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(128*9*9, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid())

    def forward(self, xb):
        return self.network(xb)


#Création d'un CNN avec 3 couches de convolution
class NewCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3,64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128*15*15, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)

    
# création de plot pour l'accuracy et les loss
def plot_accuracy(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    
def plot_loss(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');