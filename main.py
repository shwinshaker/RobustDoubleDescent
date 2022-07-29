#!./env python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from mxnet.optimizer import Signum

from copy import deepcopy
import random
import time
import datetime
import contextlib
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

# from config import get_config
from src.models import get_net
from src.pipeline import train # , train_weighted, train_reg_weighted
from src.preprocess import get_loaders, get_loaders_augment
from src.utils import Dict2Obj
from src.utils import LabelSmoothingCrossEntropy, LossFloodingCrossEntropy
from src.utils import mse_one_hot, ce_soft
from src.utils import LogCyclicLR

def train_wrap(**config):
    config = Dict2Obj(config)

    start = time.time()

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())

    # environment set
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random seed
    if config.manual_seed is None:
        config.manual_seed = random.randint(1, 10000)
    random.seed(config.manual_seed)
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)
    if config.manual_seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    ## ------------------------------- data ----------------------------------
    print('=====> Loading data..')
    trainsubids = None
    if hasattr(config, 'train_subset_path') and config.train_subset_path:
        with open(config.train_subset_path, 'rb') as f:
            trainsubids = np.load(f)

    sampleweights = dict()
    sampleweights_test = dict()
    for name_weight in ['alpha', 'lambda', 'reg', 'weps', 'num_iter']:
        path_weight = '%s_sample_path' % name_weight
        if hasattr(config, path_weight) and getattr(config, path_weight):
            if hasattr(config, 'augment') and config.augment:
                raise NotImplementedError()
            with open(getattr(config, path_weight), 'rb') as f:
                sampleweights[name_weight] = np.load(f)

    labelnoisyids = []
    if hasattr(config, 'noise_subset_path') and config.noise_subset_path:
        if hasattr(config, 'augment') and config.augment:
            raise NotImplementedError()
        with open(config.noise_subset_path, 'rb') as f:
            labelnoisyids = np.load(f)

    if hasattr(config, 'confTrack') and config.confTrack and ('TV' in config.confTrackOptions or 'KL' in config.confTrackOptions):
        assert(config.dataset in ['cifar10']), 'Distance to true distribution not supported for dataset %s' % config.dataset
        with open('%s/cifar-10h/data/cifar10h-probs.npy' % config.data_dir, 'rb') as f:
            sampleweights_test['label_distribution'] = np.load(f).astype(np.float32)

    # if hasattr(config, 'dataset_tar') and config.dataset_tar:
    if hasattr(config, 'augment') and config.augment:
        # custom dataset from local path
        loaders = get_loaders_augment(dataset=config.dataset,
                                      batch_size=config.batch_size,
                                      trainsize=config.trainsize, 
                                      trainsubids=trainsubids,
                                      data_dir=config.data_dir,
                                      config=config)
    else:
        loaders = get_loaders(dataset=config.dataset, classes=config.classes, batch_size=config.batch_size,
                              trainsize=config.trainsize, testsize=config.testsize,
                              trainsubids=trainsubids,
                              labelnoisyids=labelnoisyids,
                              weights=sampleweights,
                              testweights=sampleweights_test,
                              data_dir=config.data_dir,
                              config=config)


    ## --------------------------------- criterion ------------------------------- 
    config.reduction = 'none' # Always prevent reduction, do reduction in training script
    if hasattr(config, 'label_smoothing') and config.label_smoothing:
        criterion = LabelSmoothingCrossEntropy(reduction=config.reduction, smoothing=config.label_smoothing)
        if hasattr(config, 'loss') and config.loss != 'ce':
            raise NotImplementedError()
    elif hasattr(config, 'loss_flooding') and config.loss_flooding:
        criterion = LossFloodingCrossEntropy(reduction=config.reduction, flooding=config.loss_flooding)
        if hasattr(config, 'loss') and config.loss != 'ce':
            raise NotImplementedError()
    else:
        # criterion = nn.CrossEntropyLoss(reduction=config.reduction)
        criterion = ce_soft(reduction=config.reduction,
                            num_classes=loaders.num_classes,
                            soft_label=config.soft_label)
        if hasattr(config, 'loss'):
            if config.loss == 'ce':
                pass
            elif config.loss == 'mse':
                criterion = mse_one_hot(reduction=config.reduction,
                                        num_classes=loaders.num_classes,
                                        soft_label=config.soft_label)
            else:
                raise NotImplementedError()

    ## ---------------------------------  model ------------------------------- 
    print('=====> Initializing model..')
    if config.model_seed is not None:
        torch.manual_seed(config.model_seed)
    net = get_net(config, loaders)
    config.parallel_model = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
        config.parallel_model = True
    print(net)
    print("     Total params: %.2fM" % (sum(p.numel() for p in net.parameters())/1000000.0))

    ## -- Load weights
    if config.state_path:
        print('=====> Loading pre-trained weights..')
        assert(not config.resume), 'pre-trained weights will be overriden by resume checkpoint! Resolve this later!'
        state_dict = torch.load(config.state_path, map_location=config.device)
        net.load_state_dict(state_dict)

    ## ---------------------------------- optimizer --------------------------------
    print('=====> Initializing optimizer..')
    if config.opt.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config.lr, weight_decay=config.wd, momentum=config.momentum)
    elif config.opt.lower() == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=config.lr, weight_decay=config.wd)
    elif config.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=config.lr, weight_decay=config.wd, momentum=config.momentum)
    elif config.opt.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wd)
    else:
        raise KeyError(config.optimizer)

    ## -------------------------------------  lr scheduler ---------------------------------- 
    print('=====> Initializing scheduler..')
    scheduler = None
    if config.scheduler:
        if config.scheduler == 'multistep':
            if config.milestones:
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
            # lrs = [param_group['lr'] for param_group in scheduler.optimizer.param_groups]
        elif config.scheduler == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr=config.lr, max_lr=config.lr_max,
                                                          step_size_up=int(config.epochs/5*2),
                                                          step_size_down=int(config.epochs/5*3),
                                                          mode='triangular',
                                                          cycle_momentum=False)
        elif config.scheduler == 'log_cyclic':
            scheduler = LogCyclicLR(optimizer,
                                    base_lr=config.lr, max_lr=config.lr_max,
                                    step_size_up=int(config.epochs/5*2),
                                    step_size_down=int(config.epochs/5*3),
                                    mode='triangular',
                                    cycle_momentum=False)
        elif config.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        elif config.scheduler == 'cosine_restart':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epoch_cycle, T_mult=1)
        else:
            raise KeyError(config.scheduler)

    ## -- Load checkpoint when resuming
    config.epoch_start = 0
    if config.resume:
        print('=====> Loading state..')
        checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)
        config.epoch_start = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if config.scheduler and config.milestones:
            # note: Previously scheduler saving has a problem (get a length 2 runningtime error)
            #       try the following instead if loading previous checkpoints
            # scheduler.load_state_dict(checkpoint['scheduler'][0])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                scheduler.last_epoch = checkpoint['epoch'] - 1

    ## ------------------------------------- adversary ---------------------------------- 
    print('=====> Training..')
    train(config, net=net, loaders=loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    print('     Finished.. %.3f' % ((time.time() - start) / 60.0))


if __name__ == '__main__':

    # TODO: trace more information
    #           weight norm (std)
    #           Lipschitz (max singular value)
    #           gradient norm
    #           margin (check Bartlett.) ? How to calculate this

    # single run from config
    # args = get_config('-c config.yaml'.split())
    # # with open('checkpoints/%s/train.out' % args.config['checkpoint'],'a') as f:
    # #     with contextlib.redirect_stdout(f):
    # train_wrap(**args.config)

    # single run from json
    with open('para.json') as json_file:
        config = json.load(json_file)
        print(config)
    train_wrap(**config)


    # cerfity
    if config['adversary'] is not None:
        from script.robustCertify import robust_certify
        print('\n> --------------- Start robustness certification using auto attack ----------------')
        # model_parallel = False
        # if ',' in str(config['gpu_id']):
        #     model_parallel = True
        robust_certify(model=config['model'], depth=config['depth'], width=config['width'],
                    gpu_id=config['gpu_id'], state='best', dataset=config['dataset'],
                    mode='fast', data_dir=config['data_dir'])
        robust_certify(model=config['model'], depth=config['depth'], width=config['width'],
                    gpu_id=config['gpu_id'], state='last', dataset=config['dataset'],
                    mode='fast', data_dir=config['data_dir'])
 
    # clear up temp files
    if not config['save_checkpoint']:
        os.remove('checkpoint.pth.tar')
    if not config['save_model']:
        os.remove('best_model.pt')
        os.remove('best_model_loss.pt')
        os.remove('model.pt')

    # multiple run

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())


