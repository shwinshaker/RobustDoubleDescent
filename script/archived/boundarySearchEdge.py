#!./env python
import torch
import numpy as np

import argparse

import os
import random
import time
from itertools import combinations
from sklearn.metrics import pairwise_distances

from src.preprocess import get_loaders
from src.analyses import DataTool, get_net, get_data

def list_except(l, idx):
    idxs = [i for i in range(len(l)) if i != idx]
    return l[idxs]

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
    # n_hosts = 100 # 10
    num_points = 100 # 1000 # Try this one
    n_mesh = 41
    eps = 8
    mode = 'closest' # 'random'
    lim = (-1, 2)
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
    # inputs0, inputs1 = get_data(loaders, device=device)
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
    net = get_net(path, loaders.num_classes, loaders.n_channel, device, model=args.model)
    data_tool = DataTool(net=net,
                         length_unit=length_unit, metric=metric,
                         size=(3, 32, 32), dataset=dataset,
                         device=device)
    # delete the dataloader to free up some memory
    ## Dataloader itself doesn't take memory, but probably the dataset will
    del loaders

    # boundary search
    print('> Search boundaries')

    # indices = np.random.choice(len(inputs0), n_hosts)
    # for ic, idx in enumerate(indices):
    for ic, p0 in enumerate(inputs0):

        # p0 = inputs0[idx]

        # if mode == 'closest':
        #     distance_mat = data_tool.distance(p0, inputs1)
        #     idxs = np.argsort(distance_mat)[:num_points]
        # elif mode == 'random':
        #     idxs = np.random.choice(range(len(inputs1)), num_points)
        # else:
        #     raise KeyError

        # fix the combinations and weights for all models to be fair
        # comb_indices = random.sample(list(combinations(idxs, 2)), num_points)
        comb_indices = random.sample(list(combinations(range(num_points), 2)),
                                     num_points)

        # Searching start
        # monos = []
        # Randomly select two boundary points, check the edge boundary between them
        print('-- edge boundaries')
        save_path = os.path.join(path, "edge_boundary.pt")
        edge_boundaries = []
        for i, (idx1, idx2) in enumerate(comb_indices):
            print('--- %i -- %i -- idx: (%i - %i)     ' % (ic, i, idx1, idx2))
            edge_boundary = data_tool.get_edge_boundary(p0,
                                                        inputs1[idx1],
                                                        inputs1[idx2],
                                                        lim=lim,
                                                        n_mesh=n_mesh)
            # monos.append(mono)
            edge_boundaries.append(edge_boundary)

        # save in loop to avoid too much memory consuming
        edge_boundaries = torch.Tensor(edge_boundaries)
        if ic > 0:
            edge_boundaries_= torch.load(save_path)
            edge_boundaries = torch.cat([edge_boundaries_, edge_boundaries])
            del edge_boundaries_
        torch.save(edge_boundaries, save_path)


        ## Select one boundary points, and randomly select a point on the opposite facet
        normalized_weights = []
        for _ in range(num_points):
            weights = torch.randn(num_points - 1).to(device)
            weights /= torch.sum(weights)
            normalized_weights.append(weights)

        print('-- interior boundaries')
        save_path = os.path.join(path, "edge_boundary_random.pt")
        edge_boundaries = []
        for i, weights in enumerate(normalized_weights):
            # ib = int(i // np.sqrt(num_points))
            # print('--- %i -- %i -- idx: %i            ' % (ic, i, idxs[ib]))
            print('--- %i -- %i             ' % (ic, i))
            edge_boundary = data_tool.get_edge_boundary(p0,
                                                        # inputs1[idxs[ib]],
                                                        inputs1[i],
                                                        weights[None, :] @ list_except(inputs1, i),
                                                        lim=lim,
                                                        n_mesh=n_mesh)
            edge_boundaries.append(edge_boundary)


        # save in loop to avoid too much memory consuming
        edge_boundaries = torch.Tensor(edge_boundaries)
        if ic > 0:
            edge_boundaries_= torch.load(save_path)
            edge_boundaries = torch.cat([edge_boundaries_, edge_boundaries])
            del edge_boundaries_
        torch.save(edge_boundaries, save_path)


    print('-- Finished.. %.3f' % ((time.time() - start) / 60.0))
