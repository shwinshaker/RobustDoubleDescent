#!./env python
import torch
import torch.nn as nn
import numpy as np

import argparse

import os
import random
import time
import copy
from sklearn.metrics import pairwise_distances

from src.preprocess import get_loaders
from src.models import resnet, resnet_fixup
from src.analyses import DataTool
from src.utils import Dict2Obj

def get_net(path, num_classes, n_channel, device, model='resnet'):
    state_dict = torch.load('%s/model.pt' % path, map_location=device)
    if model == 'resnet':
        net = resnet(depth=20, num_classes=num_classes, n_channel=n_channel).to(device)
    elif model == 'resnet_fixup':
        net = resnet_fixup(depth=20, num_classes=num_classes, n_channel=n_channel).to(device)
    else:
        raise KeyError(model)
    net.load_state_dict(state_dict)
    net.eval()

    feature = copy.deepcopy(net)
    feature.fc = nn.Identity()
    model = {'net': net,
             'feature': feature,
             'classifier': copy.deepcopy(net.fc)}
    model = Dict2Obj(model)

    return model

def get_data(loaders, device, feature=None):
    inputs_dog, inputs_cat = [], []
    for ii, (ins, las) in enumerate(loaders.trainloader):
        ins, las = ins.to(device), las.to(device)
        with torch.no_grad():
            ins = feature(ins)
        print('-- %i' % ii, ins.size(), len(inputs_dog), end='\r')
        inputs_dog.append(ins[las == loaders.trainset.class_to_idx['dog']]) # preds
        inputs_cat.append(ins[las == loaders.trainset.class_to_idx['cat']]) # preds
    inputs_dog = torch.cat(inputs_dog, dim=0)
    inputs_cat = torch.cat(inputs_cat, dim=0)
    inputs_dog = inputs_dog.view(inputs_dog.size(0), -1)
    inputs_cat = inputs_cat.view(inputs_dog.size(0), -1)
    print(inputs_dog.size(), inputs_cat.size())
    return inputs_dog, inputs_cat

if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="model path")
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    parser.add_argument('-s', "--seed", default=7, type=int, help='manual seed')
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    args = parser.parse_args()

    # settings
    metric = 'l2' # inf'
    dataset = 'cifar10'
    classes = [] # ['dog', 'cat']
    path = args.path
    save_path = os.path.join(path, "boundary_feature.pt")
    num_points = 100
    eps = 8
    gpu_id = args.gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set env
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    start = time.time()

    # get dataset
    print('> Get dataset')
    loaders = get_loaders(dataset=dataset, data_dir='./data',
                          classes=classes, testsize=1000)
    
    print('> Get model')
    net = get_net(path, loaders.num_classes, loaders.n_channel, device, model=args.model)

    print('> Pick data')
    inputs_dog, inputs_cat = get_data(loaders, device=device, feature=net.feature)

    print('> Build data tool')
    length_unit = np.mean(pairwise_distances(np.vstack([inputs_dog.cpu().numpy(), inputs_cat.cpu().numpy()]),
                          metric=metric)) # chebyshev
    print(length_unit)
    data_tool = DataTool(net=net.classifier,
                         length_unit=length_unit, metric=metric,
                         # size=(3, 32, 32),
                         size=(64,),
                         dataset=dataset,
                         device=device)
    # delete the dataloader to free up some memory
    ## Dataloader itself doesn't take memory, but probably the dataset will
    del loaders

    # boundary search
    print('> Search boundaries')

    indices_dog = np.random.choice(len(inputs_dog), num_points)
    for ic, p0 in enumerate(inputs_dog[indices_dog]):
        print('-- %i' % ic)

        indices_cat = np.random.choice(range(len(inputs_cat)), num_points)

        bds = dict()
        ## Gaussian attack
        print('--- Gaussian')
        bds['gauss'] = [data_tool.boundary_search(p0, torch.randn(*p0.shape).to(device)).item() for _ in range(num_points)]

        print('--- Uniform')
        bds['uniform'] = [data_tool.boundary_search(p0, torch.rand(*p0.shape).to(device)).item() for _ in range(num_points)]

        ## Ad attack
        print('--- Ad')
        p0_ads = [data_tool.attack_data(p0, eps=eps, randomize=True, pred=False, scale=False, is_clamp=False) for _ in range(num_points)]
        # p0_ads = [data_tool.attack_data(p0, eps=eps, randomize=True, adversary='pgd', pgd_alpha=eps/4, pred=False) for _ in range(num_points)]
        bds['ad'] = [data_tool.boundary_search(p0, p0_ad - p0).item() for p0_ad in p0_ads]

        ## Pairwise
        print('--- Pairwise')
        bds['pair'] = [data_tool.boundary_search(p0, inputs_cat[idx] - p0).item() for idx in indices_cat]

        ## Convex hull
        print('--- Convex hull')
        weights = torch.rand(num_points, len(inputs_cat)).to(device)
        weights /= torch.sum(weights, dim=1, keepdims=True)
        bds['hull'] = [data_tool.boundary_search(p0, weight[None, :] @ inputs_cat - p0).item() for weight in weights]

        ## Affine hull
        print('--- Affine hull')
        weights = torch.rand(num_points, len(inputs_cat)).to(device) * 1.5 - 1.
        weights /= torch.sum(weights, dim=1, keepdims=True)
        bds['affine'] = [data_tool.boundary_search(p0, weight[None, :] @ inputs_cat - p0).item() for weight in weights]

        # save in loop to avoid too much memory consuming
        if ic > 0:
            bds_ = torch.load(save_path)
            for key in bds:
                bds[key].extend(bds_[key])
            del bds_

        torch.save(bds, save_path)

    print('-- Finished.. %.3f' % ((time.time() - start) / 60.0))
