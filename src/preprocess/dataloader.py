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

def get_loaders(dataset='cifar10', classes=None, batch_size=128,
                shuffle_train_loader=True, random_augment=True,
                trainsize=None, testsize=None, 
                trainsubids=None, testsubids=None, # for select ids
                trainextrasubids=[],
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

    # -- Select sub-classes
    if not classes:
        # save attributes
        classes = trainset.classes
        class_to_idx = trainset.class_to_idx

        # Extra subsets to evaluate training set
        trainextrasubsets = []
        if trainextrasubids:
            # assert(trainsubids is not None), 'Eval subsets are set, but train subset is not. Double check!'
            # sanity check
            if trainsubids is not None:
                trainids_ = trainsubids
            else:
                trainids_ = np.arange(len(trainset))
            for ids in trainextrasubids:
                assert(np.all(np.isin(ids, trainids_))), 'Eval subset ids is not included in the trainset ids ! Double check!'

            # select extra subids before changing trainset
            trainextrasubsets = [torch.utils.data.Subset(trainset, ids) for ids in trainextrasubids]

        trainsubids_ = None

        # Randomly select subset
        if trainsize is not None:
            assert trainsubids is None, 'selected based on ids is prohibited when size is enabled'
            assert trainsize < len(trainset), 'training set has only %i examples' % len(trainset)
            # random select subsets
            # trainids = np.arange(len(trainset))
            # trainsubids = np.random.sample(trainids, trainsize)
            trainsubids_ = np.random.choice(len(trainset), trainsize,
                                            replace=False)
            targets = [trainset.targets[i] for i in trainsubids_]
            trainset = torch.utils.data.Subset(trainset, trainsubids_)

            # recover attributes
            trainset.classes = classes
            trainset.class_to_idx = class_to_idx
            trainset.targets = targets

            # # Select a subset of training set in this training subset only for robustness validation
            # if len(testset) < len(trainset):
            #     subids = random.sample(range(len(trainset)), len(testset))
            #     trainextrasubsets.append(torch.utils.data.Subset(trainset, subids))
            # else:
            #     trainextrasubsets.append(trainset)

        # Specified subset
        if trainsubids is not None:
            assert(isinstance(trainsubids, np.ndarray))
            # targets = np.array(trainset.targets)[trainsubids].tolist()
            targets = [trainset.targets[i] for i in trainsubids]
            trainset = torch.utils.data.Subset(trainset, trainsubids)

            # recover attributes
            trainset.classes = classes
            trainset.class_to_idx = class_to_idx
            trainset.targets = targets

            # to be consistent with the variable used above
            trainsubids_ = np.array(trainsubids)

            # # Select a subset of training set in this training subset only for robustness validation
            # if len(testset) < len(trainset):
            #     subids = random.sample(range(len(trainset)), len(testset))
            #     trainextrasubsets.append(torch.utils.data.Subset(trainset, subids))
            # else:
            #     trainextrasubsets.append(trainset)

        if testsize is not None or testsubids is not None:
            raise NotImplementedError
    else:
        assert isinstance(classes, tuple) or isinstance(classes, list)
        assert all([c in trainset.classes for c in classes]), (trainset.classes, classes)

        idx_classes = [trainset.class_to_idx[c] for c in classes]
        idx_convert = dict([(idx, i) for i, idx in enumerate(idx_classes)])
        class_to_idx = dict([(c, i) for i, c in enumerate(classes)])

        # select in-class indices
        trainids = [i for i in range(len(trainset)) if trainset.targets[i] in idx_classes]
        testids = [i for i in range(len(testset)) if testset.targets[i] in idx_classes]

        # modify labels. 0-9(10 classes) -> 0-1(2 classes)
        for idx in trainids:
            trainset.targets[idx] = idx_convert[trainset.targets[idx]]
        for idx in testids:
            testset.targets[idx] = idx_convert[testset.targets[idx]]

        # select a subset if needed
        if trainsize:
            random.shuffle(trainids)
            trainids = trainids[:trainsize]
        if testsize:
            random.shuffle(testids)
            testids = testids[:testsize]
        if trainsubids is not None or testsubids is not None:
            raise NotImplementedError
        
        trainset = torch.utils.data.Subset(trainset, trainids)
        testset = torch.utils.data.Subset(testset, testids)

    if hasattr(config, 'traintest') and config.traintest:
        # select a subset of training set for robustness validation
        # ~~Do it before sampling training set so that the training accuracy is always comparable~~ ??
        if len(testset) < len(trainset):
            subids = random.sample(range(len(trainset)), len(testset))
            trainsubset = torch.utils.data.Subset(trainset, subids)
        else:
            trainsubset = trainset

    print('- training set -')
    summary(trainset, classes, class_to_idx)
    print('- test set -')
    summary(testset, classes, class_to_idx)
    if hasattr(config, 'traintest') and config.traintest:
        print('- training sub set -')
        summary(trainsubset, classes, class_to_idx)
    if trainextrasubsets:
        for subset in trainextrasubsets:
            print('- extra training sub set for evaulation -')
            summary(subset, classes, class_to_idx)

    # print(len(trainset))
    # print(len(testset))

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
    loaders.classes = classes
    loaders.class_to_idx = class_to_idx
    loaders.num_classes = len(classes)
    loaders.n_channel = n_channel
    loaders.shape = shape
    loaders.trainset = trainset
    loaders.testset = testset
    if hasattr(config, 'traintest') and config.traintest:
        loaders.trainsubset = trainsubset
    loaders.trainids = trainsubids_

    if trainextrasubsets:
        loaders.trainextrasubsets = trainextrasubsets
        loaders.trainextraloaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                                                 shuffle=False, num_workers=n_workers,
                                                                 worker_init_fn=seed_worker, generator=generator) for subset in trainextrasubsets]

    # print(classes)
    return loaders


if __name__ == '__main__':
    loaders = get_loaders() # classes=('dog', 'cat'))
    print(loaders.classes)
