# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:25:19 2023

@author: mallier
"""

#Importation des packages nécessaires et important
import torch
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

import os

from Medium_CNN import MediumCNN, MediumFit
from DC_CNN import CNNet, fit_CNN, val_CNN

#Sous dossier des données images
train_dir = './train_another'
train_damage_dir = train_dir + '/damage'
train_nodamage_dir = train_dir + '/no_damage'

validation_dir = './validation_another'
validation_damage_dir = validation_dir + '/damage'
validation_nodamage_dir = validation_dir + '/no_damage'

test_dir = './test_another'
test_damage_dir = test_dir + '/damage'
test_nodamage_dir = test_dir + '/no_damage'

##Nombre d'images par dossiers

#Equilibré en train 5000-5000
len(os.listdir(train_damage_dir))
len(os.listdir(train_nodamage_dir))

#Equilibré en validation 1000-1000
len(os.listdir(validation_damage_dir))
len(os.listdir(validation_nodamage_dir))

#Déséquilibré en test 8000-1000
len(os.listdir(test_damage_dir))
len(os.listdir(test_nodamage_dir))

##Création des jeux de données avec torchvision

#Entrainement
train = ImageFolder(train_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

#Validation
validation = ImageFolder(validation_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

#Classe des données ['damage', 'no_damage']
train.classes #Donc 0 représente les damage et 1 les no_damage

#Chargement en DataLoader avec batch 128 pour éviter les crashs en entrainement et shuffle
#pour mélanger les batchs à chaque étape de l'entrainement
train_dl = DataLoader(train, 20, shuffle = True, num_workers = 4, pin_memory = True)

val_dl = DataLoader(validation, 20, shuffle = True, num_workers = 4, pin_memory = True)

num_epochs = 10
opt_func = torch.optim.RMSprop
lr = 1e-4
model = MediumCNN()
history, model = MediumFit(num_epochs, lr, model, train_dl, val_dl, opt_func)

from torchinfo import summary

summary(model)

#DataCamp
net = CNNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4)
fit_CNN(net, train_dl, criterion, optimizer)
val_CNN(net, val_dl)




















