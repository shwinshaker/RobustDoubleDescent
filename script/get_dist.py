#!./env python
import torch
import numpy as np
import torch.nn as nn

import os
import time

from src.utils import Logger
from src.preprocess import get_loaders
from src.analyses import get_ad_examples, get_net
from src.utils import Dict2Obj
from src.utils import RobustTracker

def eval_net(net, loaders, device=None):

    ## Evaluation
    acc = 0
    loss = 0
    count = 0
    for batch_idx, (inputs, targets, _) in enumerate(loaders.testloader):
        inputs_, targets = inputs.to(device), targets.to(device)
        inputs, _ = get_ad_examples(net, inputs_, labels=targets, criterion=net._loss,
                                    adversary='pgd', eps=8, pgd_alpha=2, pgd_iter=10,
                                    dataset='cifar10', device=device)
        # inputs = inputs_
        with torch.no_grad():
            outputs = net(inputs)
            loss += net._loss(outputs, targets).item() * inputs.size(0)
            _, preds = outputs.max(1)
            acc += preds.eq(targets).sum().item()
            count += inputs.size(0)
        print('[%i / %i] %.4f %.4f' % (batch_idx + 1,
                                       len(loaders.testloader),
                                       (acc / count)*100,
                                       loss / count),
              end='\r')
    return acc / count * 100, loss / count


if __name__ == '__main__':

    gpu_id = 6
    resume = False
    width = 5
    path = 'checkpoints/adam_wrn-28-%i_gain=1_0_ad_pgd_10_alpha=1_lr=1e-04_mom=0_9_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000' % width
    random_augment = True # random augment allowed, only matters for mindist as that one is evaluated on training set

    os.chdir(path)

    ## Set device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ## Set config
    config = {'traintest': True,
              'dataset': 'cifar10',
              'batch_size': 128,
              'rbTrack': ['FGSM'],
              'rbTrackPhase': 'train',
              'rbTrackSubsize': 1000,
              'rbTrackSavebest': False,
              'device': device,
             }
    config = Dict2Obj(config)
    
    start = time.time()

    ## get loader
    dataset = 'cifar10'
    loaders = get_loaders(dataset=dataset, random_augment=random_augment, 
                          shuffle_train_loader=False,
                          data_dir='/home/chengyu/Initialization/data',
                          config=config)

    ## Robust metric log
    epoch = 0
    net = get_net('.', num_classes=loaders.num_classes, n_channel=loaders.n_channel,
                       model='wrn', depth=28, width=width, state='model-%i.pt' % epoch,
                       device=device)
    rbLog = RobustTracker(net, loaders, config, start)

    for epoch in range(0, 1000, 10):
        net_ = get_net('.', num_classes=loaders.num_classes, n_channel=loaders.n_channel,
                       model='wrn', depth=28, width=width, state='model-%i.pt' % epoch,
                       device=device)

        # switch network of rbLog
        print(epoch)
        net.load_state_dict(net_.state_dict())
        rbLog.update(epoch)

    rbLog.close()
    print('-- Finished.. %.3f mins' % ((time.time() - start) / 60.0))

