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
    classes = ['dog', 'cat']
    path = args.path
    n_hosts = 100 #  10
    num_points = 100
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
    inputs_dog, inputs_cat = get_data(loaders, device=device)

    print('> Average length as unit')
    length_unit = np.mean(pairwise_distances(np.vstack([inputs_dog.cpu().numpy(), inputs_cat.cpu().numpy()]),
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
    print('> Search angles')

    indices = np.random.choice(len(inputs_dog), n_hosts)
    for ic, idx in enumerate(indices):

        p0 = inputs_dog[idx]

        if mode == 'closest':
            distance_mat = data_tool.distance(p0, inputs_cat)
            idxs = np.argsort(distance_mat)[:num_points]
        elif mode == 'random':
            idxs = np.random.choice(range(len(inputs_cat)), num_points)
        else:
            raise KeyError

        # fix the combinations and weights for all models to be fair
        comb_indices = random.sample(list(combinations(idxs, 2)), num_points)

        # Searching start
        # monos = []
        # Randomly select two boundary points, check the edge boundary between them
        print('-- edge angles')
        save_path = os.path.join(path, "edge_angle.pt")
        edge_angles = []
        for i, (idx1, idx2) in enumerate(comb_indices):
            print('--- %i -- %i -- idx: (%i - %i)     ' % (ic, i, idx1, idx2))
            p1b, p2b = data_tool.get_boundary_points(p0, [inputs_cat[idx1], inputs_cat[idx2]])
            edgebs = data_tool.get_edge_boundary_points(p0,
                                                        inputs_cat[idx1],
                                                        inputs_cat[idx2],
                                                        n_mesh=n_mesh,
                                                        lim=lim)

            edge_angles.append(data_tool.pairwise_angle(p0 - p1b, torch.stack(edgebs) - p1b))
            edge_angles.append(data_tool.pairwise_angle(p0 - p2b, torch.stack(edgebs) - p2b)[::-1])

        # save in loop to avoid too much memory consuming
        edge_angles = torch.Tensor(edge_angles)
        if ic > 0:
            edge_angles_= torch.load(save_path)
            edge_angles = torch.cat([edge_angles_, edge_angles])
            del edge_angles_
        torch.save(edge_angles, save_path)


        ## Select one boundary points, and randomly select a point on the opposite facet
        normalized_weights = []
        for _ in range(num_points):
            weights = torch.randn(num_points - 1).to(device)
            weights /= torch.sum(weights)
            normalized_weights.append(weights)

        print('-- interior angles')
        save_path = os.path.join(path, "edge_angle_random.pt")
        edge_angles = []
        for i, weights in enumerate(normalized_weights):
            ib = int(i // np.sqrt(num_points))
            print('--- %i -- %i -- idx: %i            ' % (ic, i, idxs[ib]))
            p1 = inputs_cat[idxs[ib]]
            p2 = weights[None, :] @ list_except(inputs_cat[idxs], ib)
            p1b, p2b = data_tool.get_boundary_points(p0, [p1, p2])
            edgebs = data_tool.get_edge_boundary_points(p0, p1, p2,
                                                        n_mesh=n_mesh,
                                                        lim=lim)

            edge_angles.append(data_tool.pairwise_angle(p0 - p1b, torch.stack(edgebs) - p1b))
            # edge_angles.append(data_tools.pairwise_angle(p0 - p2b, torch.stack(edgebs) - p2b)[::-1])


        # save in loop to avoid too much memory consuming
        edge_angles = torch.Tensor(edge_angles)
        if ic > 0:
            edge_angles_= torch.load(save_path)
            edge_angles = torch.cat([edge_angles_, edge_angles])
            del edge_angles_
        torch.save(edge_angles, save_path)


    print('-- Finished.. %.3f' % ((time.time() - start) / 60.0))
