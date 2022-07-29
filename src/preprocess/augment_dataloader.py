#!./env python

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import random
import os

from . import WeightedDatasetFromDict, dataset_stats, summary

__all__ = ['get_loaders_augment']

def get_classes(dataset, data_dir):
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    else:
        raise NotImplementedError('Augmented data not prepared..')
    dataset_tmp = dataloader(root=data_dir, train=True, download=False)
    return dataset_tmp.classes, dataset_tmp.class_to_idx

def get_data_path(config):
    train_path = '%s/train_set.pt' % config.data_dir
    test_path= '%s/test_set.pt' % config.data_dir
    if config.augment in ['pgd', 'aa', 'gaussian']:
        if config.ad_aug_eps_train > 0:
            train_path = '%s/%s_train_set_eps=%i.pt' % (config.data_dir, config.augment, config.ad_aug_eps_train)
        if config.ad_aug_eps_test > 0:
            test_path = '%s/%s_test_set_eps=%i.pt' % (config.data_dir, config.augment, config.ad_aug_eps_test)
        return train_path, test_path

    if config.augment in ['mixup']:
        if config.mixup_aug_ratio_train > 0:
            train_path = '%s/%s_train_set_ratio=%g.pt' % (config.data_dir, config.augment, config.mixup_aug_ratio_train)
        if config.mixup_aug_ratio_test > 0:
            test_path = '%s/%s_test_set_ratio=%g.pt' % (config.data_dir, config.augment, config.mixup_aug_ratio_test)
        return train_path, test_path

    raise KeyError(config.augment)

def get_loaders_augment(dataset='cifar10', batch_size=128,
                        shuffle_train_loader=True, random_augment=True,
                        trainsize=None, trainsubids=None,
                        data_dir='/home/chengyu/Initialization/data',
                        n_workers=4, config=None):
    """
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    
    if dataset != 'cifar10':
        raise NotImplementedError('Augmented data not prepared..')

    train_path, test_path = get_data_path(config)
    print('train_path: %s' % train_path)
    print('test_path: %s' % test_path)
    traindata = torch.load(train_path)
    testdata = torch.load(test_path)

    # Transformations - because the saved data are already in tensor format, no need to convert to tensor here
    if random_augment:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])
    else:
        transform_train = transforms.Compose([
                transforms.Normalize(dataset_stats[dataset]['mean'],
                                     dataset_stats[dataset]['std']),])

    transform_test = transforms.Compose([
            transforms.Normalize(dataset_stats[dataset]['mean'],
                                 dataset_stats[dataset]['std']),])


    classes, class_to_idx = get_classes(dataset, data_dir)
    trainset = WeightedDatasetFromDict(traindata, classes, class_to_idx, transform=transform_train)
    testset = WeightedDatasetFromDict(testdata, classes, class_to_idx, transform=transform_test)

    # Select a subset
    trainsubids_ = None
    if trainsize is not None:
        # Randomly select subset
        assert trainsubids is None, 'selected based on ids is prohibited when size is enabled'
        assert trainsize < len(trainset), 'training set has only %i examples' % len(trainset)
        trainsubids_ = np.random.choice(len(trainset), trainsize, replace=False)

    if trainsubids is not None:
        # Specified subset
        assert(isinstance(trainsubids, np.ndarray))
        trainsubids_ = trainsubids

    if trainsubids_ is not None:
        targets = [trainset.targets[i] for i in trainsubids_]
        trainset = torch.utils.data.Subset(trainset, trainsubids_)
        # recover attributes
        trainset.classes = classes
        trainset.class_to_idx = class_to_idx
        trainset.targets = targets

    # Select a subset of training set for robustness validation
    if hasattr(config, 'traintest') and config.traintest:
        if len(testset) < len(trainset):
            subids = random.sample(range(len(trainset)), len(testset))
            trainsubset = torch.utils.data.Subset(trainset, subids)
        else:
            trainsubset = trainset

    # print some statistics of the dataset
    print('- training set -')
    summary(trainset, classes, class_to_idx)
    print('- test set -')
    summary(testset, classes, class_to_idx)
    if hasattr(config, 'traintest') and config.traintest:
        print('- training sub set -')
        summary(trainsubset, classes, class_to_idx)

    # deploy loader
    seed_worker = None
    generator = None
    if config.manual_seed is not None:
        # deterministic loader
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(config.manual_seed)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle_train_loader, num_workers=n_workers, pin_memory=True,
                                              worker_init_fn=seed_worker, generator=generator)
    if hasattr(config, 'traintest') and config.traintest:
        traintestloader = torch.utils.data.DataLoader(trainsubset, batch_size=batch_size,
                                                      shuffle=False, num_workers=n_workers, pin_memory=True,
                                                      worker_init_fn=seed_worker, generator=generator)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers, pin_memory=True,
                                             worker_init_fn=seed_worker, generator=generator)

    # integrate
    class Loaders:
        pass
    loaders = Loaders()
    loaders.trainset = trainset
    loaders.testset = testset
    if hasattr(config, 'traintest') and config.traintest:
        loaders.trainsubset = trainsubset
    loaders.trainloader = trainloader
    loaders.testloader = testloader
    if hasattr(config, 'traintest') and config.traintest:
        loaders.traintestloader = traintestloader
    # loaders.trainids = data['train']['indices']
    loaders.trainids = trainsubids_

    loaders.classes = classes
    loaders.class_to_idx = class_to_idx
    loaders.num_classes = len(classes)
    loaders.n_channel = trainset[0][0].size()[0]
    loaders.shape = trainset[0][0].size()

    return loaders

if __name__ == '__main__':
    loaders = get_loaders_augment() # classes=('dog', 'cat'))
    print(loaders.classes)
