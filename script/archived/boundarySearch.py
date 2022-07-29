#!./env python
import torch
import numpy as np

import argparse

import os
import random
import time
import copy
from sklearn.metrics import pairwise_distances

from src.preprocess import get_loaders
from src.analyses import DataTool, get_data, get_net
from src.utils import Dict2Obj

if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="model path")
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    parser.add_argument('-s', "--seed", default=7, type=int, help='manual seed')
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('-d', "--state_dict", default='model.pt', type=str, help='model state dict')
    args = parser.parse_args()

    # settings
    metric = 'l2' # inf'
    dataset = 'cifar10'
    classes = [] # ['dog', 'cat']
    path = args.path
    state_dict = args.state_dict
    save_path = os.path.join(path, "boundary_%s.pt" % state_dict)
    num_points = 100
    eps = 8
    pgd_iter = 10
    pgd_alpha = 1
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
    
    print('> Pick data')
    # sampled_classes = (('dog', ), tuple([c for c in loaders.classes if c != 'dog']))
    sampled_classes = (('dog', ), ('cat',))
    inputs0, inputs1 = get_data(loaders, device=device,
                                classes=sampled_classes,
                                n_sample=num_points)

    print('> Average length as unit')
    length_unit = np.mean(pairwise_distances(np.vstack([inputs0.cpu().numpy(), inputs1.cpu().numpy()]),
                          metric=metric)) # chebyshev
    print(length_unit)

    # get model
    print('> Get model')
    net = get_net(path, loaders.num_classes, loaders.n_channel, device, model=args.model, state=state_dict)
    data_tool = DataTool(net=net,
                         length_unit=length_unit, metric=metric,
                         size=(3, 32, 32), dataset=dataset,
                         device=device)
    # delete the dataloader to free up some memory
    ## Dataloader itself doesn't take memory, but probably the dataset will
    del loaders

    # boundary search
    print('> Search boundaries')

    # indices_dog = np.random.choice(len(inputs0), num_points)
    # for ic, p0 in enumerate(inputs0[indices_dog]):
    for ic, p0 in enumerate(inputs0):
        print('-- %i' % ic)

        # indices_cat = np.random.choice(range(len(inputs1)), num_points)

        bds = dict()
        ## Gaussian attack
        print('--- Gaussian')
        bds['gauss'] = [data_tool.boundary_search(p0, torch.randn(*p0.shape).to(device)).item() for _ in range(num_points)]

        print('--- Uniform')
        bds['uniform'] = [data_tool.boundary_search(p0, torch.rand(*p0.shape).to(device)).item() for _ in range(num_points)]

        ## Ad attack
        print('--- Ad')
        p0_ads = [data_tool.attack_data(p0, eps=eps, randomize=True, pred=False) for _ in range(num_points)]
        bds['ad'] = [data_tool.boundary_search(p0, p0_ad - p0).item() for p0_ad in p0_ads]

        print('--- Ad PGD')
        p0_ads = [data_tool.attack_data(p0, eps=eps, adversary='pgd', pgd_iter=pgd_iter, pgd_alpha=pgd_alpha, randomize=True, pred=False) for _ in range(num_points)]
        bds['ad_pgd'] = [data_tool.boundary_search(p0, p0_ad - p0).item() for p0_ad in p0_ads]

        ## Pairwise
        print('--- Pairwise')
        # bds['pair'] = [data_tool.boundary_search(p0, inputs1[idx] - p0).item() for idx in indices_cat]
        bds['pair'] = [data_tool.boundary_search(p0, p2 - p0).item() for p2 in inputs1]

        print('--- Pairwise relative')
        # bds['pair_relative'] = [data_tool.boundary_search(p0, inputs1[idx] - p0, scale=False).item() for idx in indices_cat]
        bds['pair_relative'] = [data_tool.boundary_search(p0, p2 - p0, scale=False).item() for p2 in inputs1]

        ## Convex hull
        print('--- Convex hull')
        weights = torch.rand(num_points, len(inputs1)).to(device)
        weights /= torch.sum(weights, dim=1, keepdims=True)
        bds['hull'] = [data_tool.boundary_search(p0, weight[None, :] @ inputs1 - p0).item() for weight in weights]

        ## Affine hull
        print('--- Affine hull')
        weights = torch.rand(num_points, len(inputs1)).to(device) * 1.5 - 1.
        weights /= torch.sum(weights, dim=1, keepdims=True)
        bds['affine'] = [data_tool.boundary_search(p0, weight[None, :] @ inputs1 - p0).item() for weight in weights]

        # save in loop to avoid too much memory consuming
        if ic > 0:
            bds_ = torch.load(save_path)
            for key in bds:
                bds[key].extend(bds_[key])
            del bds_

        torch.save(bds, save_path)

    print('-- Finished.. %.3f' % ((time.time() - start) / 60.0))
