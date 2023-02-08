# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:21:06 2023

@author: mallier
"""
import numpy as np

def extract_features(dataloader, model, n_sample, n_features):
    features = np.zeros(shape=(n_sample,n_features))
    batchSize = dataloader.batch_size
    labels = np.zeros(shape = (n_sample))
    i = 0
    for inputs_batch, labels_batch in dataloader:
        model.eval()
        features_batch = model(inputs_batch).detach().numpy()
        features[i * batchSize: (i + 1) * batchSize] = features_batch
        labels[i * batchSize: (i + 1) * batchSize] = labels_batch
        i += 1
        if i * batchSize >= n_sample:
            break
    return features, labels