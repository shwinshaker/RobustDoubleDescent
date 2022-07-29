#!./env python
import torch
import numpy as np
import torch.nn.functional as F

import os
import time
from collections.abc import Iterable
from collections import defaultdict

from src.utils import Logger
from src.preprocess import get_loaders, get_loaders_augment
from src.analyses import get_ad_examples, get_net
from src.utils import Dict2Obj
from src.utils import mse_one_hot


class Node:
    def __init__(self):
        self.children = dict()
        
    def _set_value(self, outputs, loss, acc, batch_size=None):
        assert(len(self.children) == 0), 'can only assign value to leaf nodes!'
        self.outputs = outputs
        self.loss = loss * batch_size
        self.acc = acc * batch_size
        self.batch_size = batch_size
        
    def _count_children(self):
        if len(self.children) == 0:
            return 1
        else:
            return sum([self.children[key]._count_children() for key in self.children])
        
    def _get_avg(self, variable, **kwargs):
        # This avg will behave differently from the naive implementation when the number of children is imbalanced
        if len(self.children) == 0:
            # reach leaf node
            if variable == 'risk':
                return np.sum((self.outputs - kwargs['targets']) ** 2)
            return getattr(self, variable)
        else:
            avg = 0
            for key in self.children:
                avg += self.children[key]._get_avg(variable, **kwargs) * self.children[key]._count_children()
            return avg / self._count_children()
        
    def _get_bias(self, targets):
        return np.sum((self._get_avg('outputs') - targets) **2)
    
    def _get_variance(self, level):
        if self.__match_level(level):
            outputs_avg = self._get_avg('outputs')
            # return np.mean([np.sum((self.children[key]._get_avg('outputs') - outputs_avg)**2) for key in self.children])
            error_list = [np.sum((self.children[key]._get_avg('outputs') - outputs_avg)**2) for key in self.children]
            return np.sum(error_list) / len(error_list) # biased
            # return np.sum(error_list) / (len(error_list) - 1) # unbiased
        else:
            avg = 0
            for key in self.children:
                avg += self.children[key]._get_variance(level) * self.children[key]._count_children()
            return avg / self._count_children()
        
    def __match_level(self, level):
        return level in list(self.keys())[0]

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        self.children[key] = value
        
    def keys(self):
        return self.children.keys()
    
    def values(self):
        return self.children.values()

    def items(self):
        return self.children.items()
    

def get_nets(config, epoch, loaders):
    nets = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
    for seg in config.seg:
        for seed in config.seed:
            for split in config.split:
                if config.kd:
                    path = 'checkpoints/sgd_mse_resnet18_kd_T=2_st=0.5_mom=0_9_sub=id_rand_10000_%i_noisesub=id_rand_10000_%i_label_noise_0.2_%i_seed=%i' % (seg, seg, split, seed)
                else:
                    # path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_%i_noisesub=id_rand_10000_%i_label_noise_0.2_%i_seed=%i' % (seg, seg, split, seed)
                    if split == 0:
                        path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_0_noisesub=id_rand_10000_0_label_noise_0.2_%i_modelSeed=%i' % (seg, seed)
                    else:
                        path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_0_noisesub=id_rand_10000_0_label_noise_0.2_%i_modelSeed=%i-%i' % (seg, seed, split)

                net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel,
                              model=config.model, depth=config.depth, width=config.width, state='model-%i.pt' % epoch,
                              device=config.device)
                net.eval()
                nets['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split] = net
    return nets

def get_outputs(net, inputs, targets, criterion):
    with torch.no_grad():
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        _, preds = outputs.max(1)
        acc = preds.eq(targets).sum() / targets.size(0)
        outputs = F.softmax(outputs, dim=1).cpu().numpy()
    return outputs, loss.item(), acc.item()

def compute_bias_variance(config, loaders, epoch=10):
    criterion = mse_one_hot(num_classes=loaders.num_classes)
    nets = get_nets(config, epoch, loaders)

    loss_avg = 0
    acc_avg = 0
    risk = 0
    bias2 = 0
    variance = defaultdict(int)
    num_ex = 0
    assert(len(config.levels) == 3), len(config.levels)
    for batch_idx, (inputs, targets, _) in enumerate(loaders.testloader):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        root = Node()
        para_dict = dict()
        for para0 in getattr(config, config.levels[0]):
            para_dict[config.levels[0]] = para0
            key0 = '%s=%i' % (config.levels[0], para0)
            root[key0] = Node()
            for para1 in getattr(config, config.levels[1]):
                para_dict[config.levels[1]] = para1
                key1 = '%s=%i' % (config.levels[1], para1) 
                root[key0][key1] = Node()
                for para2 in getattr(config, config.levels[2]):
                    para_dict[config.levels[2]] = para2
                    key2 = '%s=%i' % (config.levels[2], para2) 
                    root[key0][key1][key2] = Node()

                    net = nets['seg=%i' % para_dict['seg']]['seed=%i' % para_dict['seed']]['split=%i' % para_dict['split']]
                    outputs, loss, acc = get_outputs(net, inputs, targets, criterion)
                    root[key0][key1][key2]._set_value(outputs, loss, acc, batch_size=inputs.size(0))

        # for seg in config.seg:
        #     root['seg=%i' % seg] = Node()
        #     for seed in config.seed:
        #         root['seg=%i' % seg]['seed=%i' % seed] = Node()
        #         for split in config.split:
        #             root['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split] = Node()
        #     # for split in config.splits:
        #     #     root['seg=%i' % seg]['split=%i' % split] = Node()
        #     #     for seed in config.seeds:
        #     #         root['seg=%i' % seg]['split=%i' % split]['seed=%i' % seed] = Node()

        #             net = nets['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
        #             outputs, loss, acc = get_outputs(net, inputs, targets, criterion)
        #             root['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]._set_value(outputs, loss, acc, batch_size=inputs.size(0))
        #             # root['seg=%i' % seg]['split=%i' % split]['seed=%i' % seed]._set_value(outputs, loss, acc, batch_size=inputs.size(0))

        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).float().cpu().numpy()
        loss_avg += root._get_avg('loss')
        acc_avg += root._get_avg('acc')
        risk += root._get_avg('risk', targets=targets_one_hot) 
        bias2 += root._get_bias(targets=targets_one_hot) 
        for level in config.levels:
            variance[level] += root._get_variance(level=level)
        num_ex += inputs.size(0)

    loss_avg /= num_ex
    acc_avg /= num_ex
    risk /= num_ex
    bias2 /= num_ex
    for key in variance:
        variance[key] = variance[key] / num_ex
    return risk, bias2, variance, loss_avg, acc_avg


if __name__ == '__main__':

    config = {
        'loss_type': 'square',
        'seg': [0, 1, 2], # [0, 1], # , 2],
        'seed': [7, 8], # , 9],
        'split': [0, 1, 2], # [0, 1, 2, 3, 4],
        'kd': False,
        'levels': ['seg', 'split', 'seed'], # split first
        # 'levels': ['seg', 'split', 'seed'], # seed first

        'model': 'resnet', # wrn
        'depth': 18,
        'width': 16, # 5
        'resume': False, # True
        'gpu_id': 0,
        'dataset': 'cifar10',
        'data_dir': '/home/chengyu/Initialization/data',
        'epoch_start': 0,
        'epoch_end': 250,
    }
    config = Dict2Obj(config)

    ## Set device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(config.device)
    
    start = time.time()

    ## get loader
    loaders = get_loaders(dataset=config.dataset,
                          random_augment=False,
                          shuffle_train_loader=False,
                          data_dir=config.data_dir)

    ### computer bias variance
    file_name = 'log_%s_resnet-%i-%i' % (config.loss_type, config.depth, config.width)
    for key in config.levels:
        file_name += '_%s=%s' % (key, '-'.join([str(para) for para in getattr(config, key)]))
    if config.kd:
        file_name += '_kd'
    # file_name += '_split_first_test'
    file_name += '_modelseed'

    save_dir = 'bias_variance'
    logger = Logger('tmp/%s/%s.txt' % (save_dir, file_name), title='log', resume=config.resume)
    base_names = ['Epoch']
    metrics = ['Risk', 'Bias']
    metrics += ['Variance-%s' % level for level in config.levels]
    metrics += ['Loss', 'Acc']
    # metrics += ['Loss%i' % i for i in range(config.num_splits)]
    # metrics += ['Acc%i' % i for i in range(config.num_splits)]
    logger.set_names(base_names + metrics)

    for epoch in range(config.epoch_start, config.epoch_end, 10):
        risk, bias2, variances, loss, acc = compute_bias_variance(config, loaders, epoch=epoch)
        str_cpl = '\n[%i] Risk: %.4f Bias: %.4f Variance: %.4f Loss: %.4f Acc: %.4f'
        print(str_cpl % (epoch, risk, bias2, variances['split'], loss, acc))
        logs = [epoch, risk, bias2]
        logs.extend([variances[level] for level in config.levels])
        logs.extend([loss, acc])
        logger.append(logs)

    print('-- Finished.. %.3f mins' % ((time.time() - start) / 60.0))


