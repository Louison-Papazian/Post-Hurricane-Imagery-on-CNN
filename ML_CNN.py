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
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(train_features, train_labels)
y_pred = classifier.predict(test_features)
accuracy_score(test_labels, y_pred)

# ---- Random forest ----
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(train_features, train_labels)
y_pred = classifier.predict(test_features)
accuracy_score(test_labels, y_pred)

# ---- LightGBM ----
import lightgbm as lgb
classifier = lgb.LGBMClassifier()
classifier.fit(train_features, train_labels)
y_pred = classifier.predict(test_features)
accuracy_score(test_labels, y_pred)

