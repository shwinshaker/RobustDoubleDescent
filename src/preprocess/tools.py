#!./env python

import torch

import numpy as np
import random
import os

from collections import Counter
from collections.abc import Iterable

## Custom dataset with per-sample weight
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weights=None, config=None):
        assert(isinstance(dataset, torch.utils.data.Dataset))
        if weights is None:
            self.weights = {}
        else:
            for key in weights:
                assert(len(weights[key]) == len(dataset)), (key, len(weights[key]), len(dataset))
            self.weights = weights
        self.dataset = dataset

        # save attributes
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.targets = dataset.targets

        self.config = config
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        # Your transformations here (or set it in CIFAR10)
        # weight = self.weights[index]
        weight = dict([(key, self.weights[key][index]) for key in self.weights])
        weight['index'] = index

        # sanity check
        if 'alpha' in weight and weight['alpha'] < 0.2:
            if 'reg' in weight:
                assert(weight['reg'] == 0), 'adding label smoothing to clean loss will cause false robustness! Index: %i, ls: %.2f, alpha: %.2f' % (index, weight['reg'], weight['alpha'])
            else:
                assert(not hasattr(self.config, 'label_smoothing') or self.config.label_smoothing == 0), 'adding label smoothing to clean loss will cause false robustness! Index: %i, ls: %.2f, alpha: %.2f' % (index, self.config.label_smoothing, weight['alpha'])
        
        return data, target, weight

    def __len__(self):
        return len(self.dataset)


## Custom dataset from local saved path
class WeightedDatasetFromDict(torch.utils.data.Dataset):
    def __init__(self, data, classes, class_to_idx, transform=None, config=None):

        self.inputs = data['inputs']
        self.targets = data['targets']
        self.indices = data['indices']
        self.weights = {} # weights
        if 'targets2' in data:
            self.weights['targets2'] = data['targets2'] # borrow weights to deliver targets2
        self.transform = transform

        self.classes = classes
        self.class_to_idx = class_to_idx

        self.config = config

    def __getitem__(self, index):
        inpt, target = self.inputs[index], self.targets[index]
        if self.transform is not None:
            inpt = self.transform(inpt)

        weight = dict([(key, self.weights[key][index]) for key in self.weights])
        weight['index'] = self.indices[index]

        return inpt, target, weight

    def __len__(self):
        return len(self.inputs)


## Dataset with custom transform
class DatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        inpt, target = self.dataset[index][0], self.dataset[index][1]
        if self.transform is not None:
            inpt = self.transform(inpt)

        return inpt, target
    
    @property
    def classes(self):
        return self.dataset.classes
    
    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx
    
    @property
    def data(self):
        return self.dataset.data   
    
    @property
    def targets(self):
        return self.dataset.targets    

    def __len__(self):
        return len(self.dataset)


## Summarize a dataset
def summary(dataset, classes, class_to_idx):
    print('shape: ', dataset[0][0].size())
    print('size: %i' % len(dataset))
    print('num classes: %i' % len(classes))
    print('---------------------------')
    def singleton(label):
        if isinstance(label, Iterable):
            return label.argmax()
        else:
            return label
    if len(dataset[0]) == 2:
        d = dict(Counter([classes[singleton(label)] for _, label in dataset]).most_common())
    else:
        d = dict(Counter([classes[singleton(label)] for _, label, _ in dataset]).most_common())
    for c in classes:
        if c in d:
            print('%s: %i' % (c, d[c]))
        else:
            print('%s: %i' % (c, 0))
    print('\n')


## label noise
def rand_target_exclude(classes, target):
    classes_ = [c for c in classes if c != target]
    return np.random.choice(classes_)

def add_label_noise(dataset, ids=None, ratio=None, num_classes=10, note=None):
    classes = list(range(num_classes))
    if ids is None:
        assert(ratio is not None)
        ids = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        save_path = 'id_label_noise_rand_ratio=%g' % ratio
        if note is not None:
            save_path += '_%s' % note
        save_path += '.npy'
        with open(save_path, 'wb') as f:
            np.save(f, ids)
        
    for i in ids:
        dataset.targets[i] = rand_target_exclude(classes, dataset.targets[i])
