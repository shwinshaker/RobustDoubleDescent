#!./env python

import torch
import torchvision.transforms as transforms

import numpy as np
import random
import os

from . import dataset_stats, get_dataloader
from . import WeightedDataset, summary
from . import add_label_noise

__all__ = ['get_loaders']

def add_trainsubids(trainsubids, ids):
    assert(not np.any(np.intersect1d(trainsubids, ids))), 'subset ids intersects!'
    return np.concatenate([trainsubids, ids])

def get_loaders(dataset='cifar10', classes=None, batch_size=128,
                shuffle_train_loader=True, random_augment=True,
                trainsize=None, testsize=None, 
                trainsubids=None, testsubids=None, # for select ids
                labelnoisyids=[], # training set only
                weights={}, testweights={},
                data_dir='/home/jingbo/chengyu/Initialization/data',
                download=False, config=None):
    """
        Property: support selection of classes
    """

    n_workers = 4
    if hasattr(config, 'n_workers'):
        n_workers = config.n_workers

    # -- Define dataset loader
    dataloader = get_dataloader(dataset, config)

    # -- Transformation
    if dataset == 'mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(), # To tensor implicitly applies a min-max scaler, such that the range is [0, 1]
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    elif dataset == 'dmnist':
        transform_train = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                lambda tensor: tensor[0, :, :].unsqueeze(0),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])

        transform_test = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                lambda tensor: tensor[0, :, :].unsqueeze(0),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    elif dataset in ['cifar10', 'cifar100', 'cifar10h', 'svhn']:
        if random_augment:
            transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'],
                                         dataset_stats[dataset]['std']),])
        else:
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'],
                                         dataset_stats[dataset]['std']),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    elif dataset == 'tiny-imagenet':
        if random_augment:
            transform_train = transforms.Compose([
                    transforms.RandomRotation(20),
                    # transforms.RandomCrop(64, padding=4), # 32x32 -> 32x32
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'],
                                         dataset_stats[dataset]['std']),])
        else:
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats[dataset]['mean'],
                                         dataset_stats[dataset]['std']),])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    else:
        raise KeyError('dataset: %s ' % dataset)

    # -- Get dataset
    trainset = dataloader(root=data_dir,
                          train=True,
                          download=download,
                          transform=transform_train)

    if len(labelnoisyids) > 0:
        add_label_noise(trainset, ids=labelnoisyids, ratio=None)
    if hasattr(config, 'trainnoisyratio') and config.trainnoisyratio:
        assert(not (len(labelnoisyids) > 0)), 'double noise standards set!'
        add_label_noise(trainset, ids=None, ratio=config.trainnoisyratio)

    trainset = WeightedDataset(trainset, weights=weights, config=config)

    testset = dataloader(root=data_dir,
                         train=False,
                         download=download,
                         transform=transform_test)

    if hasattr(config, 'testnoisyratio') and config.testnoisyratio:
        add_label_noise(testset, ratio=config.testnoisyratio, note='test')

    # just add the index, no additional weights allowed
    print(testweights.keys())
    testset = WeightedDataset(testset, weights=testweights)

    # save attributes
    classes = trainset.classes
    class_to_idx = trainset.class_to_idx

    # select a subset of training set for robustness validation
    # ~~Do it before sampling training set so that the training accuracy is always comparable~~ ??
    if hasattr(config, 'traintest') and config.traintest:
        if len(testset) < len(trainset):
            subids = random.sample(range(len(trainset)), len(testset))
            trainsubset = torch.utils.data.Subset(trainset, subids)
        else:
            trainsubset = trainset

    # - select training subset
    trainids = np.arange(len(trainset))

    # Randomly select subset
    if trainsize is not None:
        assert trainsubids is None, 'selected based on ids is prohibited when size is enabled'
        assert trainsize < len(trainset), 'training set has only %i examples' % len(trainset)
        trainids = np.random.choice(trainids, trainsize, replace=False)
        np.save('id_train_%s_size=%i.npy' % (dataset, trainsize), trainids)

    # Specified subset
    if trainsubids is not None:
        assert(isinstance(trainsubids, np.ndarray))
        trainids = np.array(trainsubids)

    # -- train validation split
    if hasattr(config, 'valsize') and config.valsize:
        if config.valsize < 1: # treat as ratio
            config.valsize = int(config.valsize * len(trainset))
        assert config.valsize < len(trainids), 'training set has only %i examples' % len(trainids)
        # select validation set
        rng = np.random.default_rng(7) # fixed seed
        valids = rng.choice(trainids, config.valsize, replace=False)
        np.save('id_val_%s_size=%i.npy' % (dataset, config.valsize), valids)

        # make val set
        targets = [trainset.targets[i] for i in valids]
        valset = torch.utils.data.Subset(trainset, valids)
        valset.classes = classes
        valset.class_to_idx = class_to_idx
        valset.targets = targets

        # rest of the training set
        trainids = np.setdiff1d(trainids, valids)

    if hasattr(config, 'valsize2') and config.valsize2: # for test
        if config.valsize2 < 1: # treat as ratio
            config.valsize2 = int(config.valsize2 * len(trainset))
        assert config.valsize2 < len(trainids), 'training set has only %i examples left' % len(trainids)
        rng = np.random.default_rng(6)
        valids = rng.choice(trainids, config.valsize2, replace=False)
        np.save('id_val_%s_size=%i.npy' % (dataset, config.valsize2), valids)

        # rest of the training set
        trainids = np.setdiff1d(trainids, valids)

    # - make train subset
    targets = [trainset.targets[i] for i in trainids]
    trainset = torch.utils.data.Subset(trainset, trainids)
    trainset.classes = classes
    trainset.class_to_idx = class_to_idx
    trainset.targets = targets

    print('- training set -')
    summary(trainset, classes, class_to_idx)
    print('- test set -')
    summary(testset, classes, class_to_idx)
    if hasattr(config, 'traintest') and config.traintest:
        print('- training sub set -')
        summary(trainsubset, classes, class_to_idx)
    if hasattr(config, 'valsize') and config.valsize:
        print('- validation set -')
        summary(valset, classes, class_to_idx)

    # ----------
    # deploy loader
    seed_worker = None
    generator = None
    if hasattr(config, 'manual_seed') and config.manual_seed is not None:
        # deterministic loader
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(config.manual_seed)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle_train_loader, num_workers=n_workers,
                                              worker_init_fn=seed_worker, generator=generator)
    if hasattr(config, 'traintest') and config.traintest:
        traintestloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
                                                      shuffle=False, num_workers=n_workers,
                                                      worker_init_fn=seed_worker, generator=generator)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers,
                                             worker_init_fn=seed_worker, generator=generator)

    if hasattr(config, 'valsize') and config.valsize:
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, num_workers=n_workers,
                                                worker_init_fn=seed_worker, generator=generator)


    # integrate
    n_channel = trainset[0][0].size()[0]
    shape = trainset[0][0].size()

    class Loaders:
        pass
    loaders = Loaders()
    loaders.trainloader = trainloader
    loaders.testloader = testloader
    if hasattr(config, 'traintest') and config.traintest:
        loaders.traintestloader = traintestloader
    if hasattr(config, 'valsize') and config.valsize:
        loaders.valloader = valloader
    loaders.classes = classes
    loaders.class_to_idx = class_to_idx
    loaders.num_classes = len(classes)
    loaders.n_channel = n_channel
    loaders.shape = shape
    loaders.trainset = trainset
    loaders.testset = testset
    if hasattr(config, 'traintest') and config.traintest:
        loaders.trainsubset = trainsubset
    if hasattr(config, 'valsize') and config.valsize:
        loaders.valset = valset
    loaders.trainids = trainsubids_

    # print(classes)
    return loaders


if __name__ == '__main__':
    loaders = get_loaders() # classes=('dog', 'cat'))
    print(loaders.classes)
