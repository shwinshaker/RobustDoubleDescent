#!./env python

import torch
import numpy as np
import os
import argparse

# import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from src.adversary import AAAttacker
from src.analyses import get_net
from src.utils import str2bool
from src.preprocess import get_loaders
from src.utils import Dict2Obj

def robust_certify(model, depth, width, state='last',
                   normalize=True, virtual=False, eps=8,
                   path='.', log_path=None, gpu_id='0', # sample=1000, seed=7,
                   mode='standard',
                   dataset='cifar10', data_dir='/home/chengyu/Initialization/data'):

    ## Current setting: evaluate on a random subset of 1000 (fixed during training)
    ## Fast setting for epoch-wise evaluation: same as above but use agpd-t only
    ## Leaderboard evalulation setting: n_ex=10000, i.e. use the entire testset

    # standard: all four attack, entire test set
    # fast: first two attack, entire test set
    assert(mode in ['standard', 'fast', 'fab', 'fab-t', 'square'])

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if log_path is None:
        log_path = 'log_certify_%s' % state
        if mode != 'standard':
            log_path += '_%s' % mode
        if eps != 8:
            log_path + '_eps=%g' % eps
        log_path += '.txt'

    
    print('>>>>>>>>>>> get loader..')
    loader_ = get_loaders(dataset=dataset,
                          data_dir=data_dir,
                          config=Dict2Obj({'soft_label': False,
                                           'soft_label_test': False,
                                           'random_augment': True}))

    print('>>>>>>>>>>> get net..')
    if state == 'last':
        model_state = 'model.pt'
    elif state == 'best':
        model_state = 'best_model.pt'
    elif str(state).isdigit(): 
        model_state = 'model-%i.pt' % int(state)
    elif any(char.isdigit() for char in state):
        phase, idx = state.split('-')
        model_state = '%s_model-%s.pt' % (phase, idx)
    else:
        model_state = '%s.pt' % state

    config = Dict2Obj({'model': model,
                       'depth': depth,
                       'width': width,
                       'device': device})

    net = get_net(path,
                  num_classes=loader_.num_classes,
                  n_channel=loader_.n_channel,
                  feature=None,
                  model=model,
                  depth=depth,
                  width=width,
                  state=model_state,
                  device=device)

    print('>>>>>>>>>>> start evaluating..')
    attacker = AAAttacker(net=net,
                          eps=eps,
                          normalize=normalize,
                          mode=mode,
                          virtual=virtual,
                          path=path,
                          log_path=log_path,
                          dataset=dataset,
                          device=device,
                          data_dir=data_dir)
    attacker.evaluate()

    print('>>>>>>>>>>> Done.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cifar10', type=str, help='dataset')
    parser.add_argument("--eps", default=8, type=float, help='perturbation radius')
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=20, type=int, help='model depth')
    parser.add_argument('--width', default=64, type=int, help='model width')
    parser.add_argument("--norm", type=str2bool, nargs='?', const=True, default=False, help="normalized inputs?")
    parser.add_argument("--virt", type=str2bool, nargs='?', const=True, default=False, help="use model's own prediction as label?")
    parser.add_argument("--mode", default='standard', type=str, help="eval mode")
    parser.add_argument('-d', "--state", default='last', type=str, help='model state')
    parser.add_argument("-p", "--path", type=str, help="model path")
    parser.add_argument("-lp", "--log_path", type=str, help="log path")
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    args = parser.parse_args()

    robust_certify(model=args.model, depth=args.depth, width=args.width, state=args.state,
                   normalize=args.norm, virtual=args.virt,
                   dataset=args.dataset,
                   mode=args.mode, eps=args.eps,
                   path=args.path, log_path=args.log_path, gpu_id=args.gpu)
