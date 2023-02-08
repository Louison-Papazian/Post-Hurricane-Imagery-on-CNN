# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:43:07 2023

@author: mallier

Implémentation de modèles sklearn
"""
import torch
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torchsummary

from utils import extract_features

train_dir = './train_another'
validation_dir = './validation_another'
test_dir = './test_another'

train = ImageFolder(train_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

validation = ImageFolder(validation_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

trainLoader = DataLoader(train, 20, shuffle = True, num_workers = 4, pin_memory = True)
validationLoader = DataLoader(validation, 20, shuffle = True, num_workers = 4, pin_memory = True)

# ---- Importation d'un modèle ----
model = torchvision.models.resnet18(pretrained=True)
torchsummary.summary(model, (3,150,150))
#On récupère le nombre output du modèle

# ---- Extraction des features ----
train_features, train_labels = extract_features(trainLoader, model, 10000, 1000)
test_features, test_labels = extract_features(validationLoader, model, 2000, 1000)

#Taille (Normalement n_sample, n_features)
train_features.shape

#Classes
train_labels

# ---- Entrainement d'un modèle XGB a partir des features de sortie du modèle resnet18 ----
import xgboost as xgb
dtrain = xgb.DMatrix(train_features, label=train_labels)
dtest = xgb.DMatrix(test_features, label=test_labels)

#Parametre de base
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

#Parametre d'evaluation
evallist = [(dtrain, 'train'), (dtest, 'eval')]

#Nb boucle
num_round = 12

#Train
bst = xgb.train(param, dtrain, num_round, evallist)
