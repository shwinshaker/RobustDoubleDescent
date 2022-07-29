#!./env python
import torch
import torch.nn as nn
import numpy as np

import argparse
from collections import defaultdict

import os
import random
import time
from sklearn.metrics import pairwise_distances

from src.preprocess import get_loaders
from src.models import resnet, resnet_fixup
from src.analyses import DataTool, get_data, get_net
from src.utils import Dict2Obj

def normalize(vec, metric=2):
    metric = {2: 'fro', 'inf': float('inf')}.get(metric)
    if len(vec.shape) == 1:
        return vec / torch.norm(vec, metric)
    return vec / torch.norm(vec, metric, dim=1, keepdim=True)

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

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
    classes = ['dog', 'cat']
    path = args.path
    save_path = os.path.join(path, "variance_1vsAll.pt")
    # num_hosts = 100
    # num_points = 200
    num_points = 50
    randomize = True # False
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
    
    # get model
    print('> Get model')
    net = get_net(path, loaders.num_classes, loaders.n_channel, device, model=args.model, feature=True)

    print('> Pick data')
    sampled_classes = (('dog', ), tuple([c for c in loaders.classes if c != 'dog']))
    inputs0, inputs1 = get_data(loaders, device,
                                classes=sampled_classes,
                                n_sample=num_points)
    # inputs0_feat, inputs1_feat = get_data(loaders, device,
    #                                       feature=net.feature,
    #                                       classes=sampled_classes,
    #                                       n_sample=num_points)

    print('> Average length as unit')
    length_unit = np.mean(pairwise_distances(np.vstack([inputs0.cpu().numpy(), inputs1.cpu().numpy()]),
                          metric=metric)) # chebyshev
    print(length_unit)

    data_tool = DataTool(net=net.net,
                         length_unit=length_unit, metric=metric,
                         size=(3, 32, 32), dataset=dataset,
                         device=device)
    # delete the dataloader to free up some memory
    ## Dataloader itself doesn't take memory, but probably the dataset will
    del loaders

    
    print('> Search boundaries')

    def input_to_feature(p):
        with torch.no_grad():
            feats = net.feature(data_tool.data_to_torch(p))
        return feats.view(feats.size(0), -1)
    
    def get_boundary(p0, vecs):
        return np.array([data_tool.boundary_search(p0, vec).item() for vec in vecs])
    
    def get_variance(vecs, contrasts):
        return torch.std(normalize(vecs) @ contrasts.T, dim=1).cpu().numpy().ravel()
    
    def get_angle(vecs, contrasts):
        return np.mean(data_tool.pairwise_angle(vecs, contrasts), axis=1)


    # indices = np.random.choice(len(inputs0), num_hosts)
    # rand_inds = np.random.choice(len(inputs1), num_points)
    # for ic, idx in enumerate(indices):
    inputs0_feat, inputs1_feat = input_to_feature(inputs0), input_to_feature(inputs1)
    for ic, p0 in enumerate(inputs0):

        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        print('---- %i' % ic)
        # p0 = inputs0[idx]
        p0_feat = input_to_feature(p0)
        contrasts = inputs1 - p0
        contrasts_feat = inputs1_feat - p0_feat
            
        ## gaussian
        ### Use gaussian as random directions, norm scaled to average length
        gau_points = p0 + normalize(torch.randn(num_points, 3072).to(device)) * length_unit
        stats['bd']['gaussian']['input'].append(get_boundary(p0, gau_points - p0))
        stats['variance']['gaussian']['input'].append(get_variance(gau_points - p0, contrasts))
        stats['angle']['gaussian']['input'].append(get_angle(gau_points - p0, contrasts))        
        
        gau_points_feat = input_to_feature(gau_points)
        stats['variance']['gaussian']['feature'].append(get_variance(gau_points_feat - p0_feat, contrasts_feat))
        stats['angle']['gaussian']['feature'].append(get_angle(gau_points_feat - p0_feat, contrasts_feat))
        
        ## pairwise points
        stats['bd']['pair']['input'].append(get_boundary(p0, inputs1 - p0))
        stats['variance']['pair']['input'].append(get_variance(inputs1 - p0, contrasts))
        stats['angle']['pair']['input'].append(get_angle(inputs1 - p0, contrasts))        
                
        stats['variance']['pair']['feature'].append(get_variance(inputs1_feat - p0_feat, contrasts_feat))
        stats['angle']['pair']['feature'].append(get_angle(inputs1_feat - p0_feat, contrasts_feat))
        
        ## Ads
        p0_ads = torch.cat([data_tool.attack_data(p0, eps=8, randomize=randomize, pred=False) for _ in range(num_points)])
        stats['bd']['ad']['input'].append(get_boundary(p0, p0_ads - p0))
        stats['variance']['ad']['input'].append(get_variance(p0_ads - p0, contrasts))
        stats['angle']['ad']['input'].append(get_angle(p0_ads - p0, contrasts))
                
        p0_ads_feat = input_to_feature(p0_ads)
        stats['variance']['ad']['feature'].append(get_variance(p0_ads_feat - p0_feat, contrasts_feat))
        stats['angle']['ad']['feature'].append(get_angle(p0_ads_feat - p0_feat, contrasts_feat))
        
        ## Ad PGD
        p0_ads = torch.cat([data_tool.attack_data(p0, eps=8, adversary='pgd', pgd_iter=5, pgd_alpha=2, randomize=randomize, pred=False) for _ in range(num_points)])
        stats['bd']['ad_pgd']['input'].append(get_boundary(p0, p0_ads - p0))
        stats['variance']['ad_pgd']['input'].append(get_variance(p0_ads - p0, contrasts))
        stats['angle']['ad_pgd']['input'].append(get_angle(p0_ads - p0, contrasts))
                
        p0_ads_feat = input_to_feature(p0_ads)
        stats['variance']['ad_pgd']['feature'].append(get_variance(p0_ads_feat - p0_feat, contrasts_feat))
        stats['angle']['ad_pgd']['feature'].append(get_angle(p0_ads_feat - p0_feat, contrasts_feat))
    
        # convert to regular dicts
        stats = ddict2dict(stats)

        # save in loop to avoid too much memory consuming
        if ic > 0:
            stats_ = torch.load(save_path)
            for t in stats:
                for k in stats[t]:
                    for p in stats[t][k]:
                        stats[t][k][p].extend(stats_[t][k][p])
            del stats_

        torch.save(stats, save_path)

    print('-- Finished.. %.3f' % ((time.time() - start) / 60.0))
