#!./env python

import torch
import torchvision.datasets as datasets

import numpy as np
import random
import os
import pickle

from . import DatasetWithTransform

__all__ = ['dataset_stats', 'get_dataloader', 'CIFAR10H']

dataset_stats = {'cifar10': {'mean': (0.49139968, 0.48215841, 0.44653091),
                             'std': (0.24703223, 0.24348513, 0.26158784)},
                 'cifar100': {'mean': (0.50707516, 0.48654887, 0.44091784),
                              'std': (0.26733429, 0.25643846, 0.27615047)},
                 'svhn': {'mean': (0.5, 0.5, 0.5),
                          'std': (0.5, 0.5, 0.5)},
                 'cifar10h': {'mean': (0.49139968, 0.48215841, 0.44653091),
                             'std': (0.24703223, 0.24348513, 0.26158784)},
                 'mnist': {'mean': (0.1306605,),
                           'std': (0.3081078,)},
                 'dmnist': {'mean': (0.05,),
                            'std': (0.1646,)},
                 'tiny-imagenet': {'mean': (0.4802, 0.4481, 0.3975),
                                   'std': (0.2302, 0.2265, 0.2262)},
                 }


def get_dataloader(dataset, config=None):
    if hasattr(config, 'aux_data') and config.aux_data is not None:
        assert(dataset == 'cifar10'), 'auxiliary data not implemented for %s' % dataset
    if dataset == 'mnist':
        def dataloader(root='.',
                       train=True,
                       download=False,
                       transform=None):
            dataset = datasets.MNIST(root=root,
                                     train=train,
                                     download=download,
                                     transform=transform)
            # targets in mnist are torch tensors, convert to number
            dataset.targets = [t.item() for t in dataset.targets]
            return dataset
    elif dataset == 'dmnist':
        def dataloader(root='.',
                       train=True,
                       download=False,
                       transform=None):
            if download:
                raise NotImplementedError('Please download multi-digit mnist manually..')
            if train:
                dataset = datasets.ImageFolder(os.path.join(root, 'double_mnist_seed_123_image_size_64_64', 'train'),
                                               transform=transform)
            else:
                dataset = datasets.ImageFolder(os.path.join(root, 'double_mnist_seed_123_image_size_64_64', 'val'),
                                               transform=transform)

            # modify the label to a random one
            classes = [c for c in dataset.classes]
            def target_transform(t):
                rand_int = np.random.randint(2)
                return int(classes[t][rand_int])
            
            dataset.samples = [(d, target_transform(t)) for d, t in dataset.samples]
            dataset.targets = [t for _, t in dataset.samples]
            dataset.classes = [str(t) for t in sorted(set(dataset.targets))]
            dataset.class_to_idx = dict([(str(t), t) for t in sorted(set(dataset.targets))])
            return dataset
    elif dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        if hasattr(config, 'aux_data') and config.aux_data is not None:
            def dataloader(root='.',
                           train=True,
                           download=False,
                           transform=None):

                if not train:
                    return datasets.CIFAR10(root=root,
                                            train=train,
                                            download=download,
                                            transform=transform)

                # load without transform
                dataset = datasets.CIFAR10(root=root,
                                           train=train,
                                           download=download)

                # append auxiliary data
                aux_path = os.path.join(config.data_dir, config.aux_data)
                print("Loading auxiliary data from %s" % aux_path)
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                dataset.data = np.concatenate((dataset.data, aux['data']), axis=0)
                dataset.targets.extend(aux['extrapolated_targets'])

                # process transformations
                dataset = DatasetWithTransform(dataset, transform)
                return dataset

    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100

    elif dataset == 'svhn':
        def dataloader(root='.',
                       train=True,
                       download=False,
                       transform=None):
            if train:
                split = 'train'
            else:
                split = 'test'
            dataset = datasets.SVHN(root=os.path.join(root, 'svhn'),
                                    split=split,
                                    transform=transform,
                                    download=download)
            dataset.classes = np.unique(dataset.labels).tolist()
            dataset.class_to_idx = dict(zip(dataset.classes, range(len(dataset.classes))))
            dataset.targets = dataset.labels
            return dataset

    elif dataset == 'tiny-imagenet':
        # custom dataloader for tiny-imagenet
        def dataloader(root='.',
                       train=True,
                       download=False,
                       transform=None):
            if download:
                raise NotImplementedError('Please download tiny-imagenet manually..')
            if train:
                return datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=transform)
            else:
                return datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'val'), transform=transform)

    elif dataset == 'cifar10h':
        def dataloader(root='.', train=True, download=False, transform=None):
            return CIFAR10H(root=root, train=train, download=download, transform=transform, softlabel=config.soft_label)

    else:
        raise KeyError('dataset: %s ' % dataset)
    return dataloader
    

class CIFAR10H(torch.utils.data.Dataset):
    def __init__(self, root='/home/jingbo/chengyu/Initialization/data',
                 train=True, download=False, transform=None, testfraction=0.1, softlabel=True):
        
        self.dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        self.targets = np.load('%s/cifar-10h/data/cifar10h-probs.npy' % root).astype(np.float32)
        if not softlabel:
            self.targets = self.targets.argmax(axis=1)
        
        size = len(self.dataset)
        trainsize = int((1 - testfraction) * size)
        
        ids = np.arange(size)
        rng = np.random.default_rng(7)
        rng.shuffle(ids)
        if train:
            self.ids = ids[:trainsize]
        else:
            self.ids = ids[trainsize:]
        
        self.num_classes = len(self.dataset.classes)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __getitem__(self, index):
        id_org = self.ids[index]
        return self.dataset[id_org][0], self.targets[id_org]

    def __len__(self):
        return len(self.ids)