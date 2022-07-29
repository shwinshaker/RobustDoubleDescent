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

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    tuples = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return ['-'.join(tup) for tup in tuples]

def get_nets(config, loaders, epoch=1000, width=16, suffix='-1'):
    nets = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
    for seg in config.segs:
        for seed in config.seeds:
            for split in config.splits:
                if config.kd:
                    # path = 'checkpoints/sgd_mse_resnet18_kd_T=2_st=0.5_mom=0_9_sub=id_rand_10000_%i_noisesub=id_rand_10000_%i_label_noise_0.2_%i_seed=%i' % (seg, seg, split, seed)
                    if width == 16:
                        path = 'checkpoints/cifar100_sgd_mse_resnet18_kd_T=2_st=0.5_mom=0_9_sub=id_rand_10000_%i_modelSeed=%i_%i' % (seg, seed, split)
                    else:
                        path = 'checkpoints/cifar100_sgd_mse_resnet18_width=%i_kd_T=2_st=0.5_mom=0_9_sub=id_rand_10000_%i_modelSeed=%i_%i' % (width, seg, seed, split)
                else:
                    # path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_%i_noisesub=id_rand_10000_%i_label_noise_0.2_%i_seed=%i' % (seg, seg, split, seed)
                    if width == 16:
                        path = 'checkpoints/cifar100_sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_%i_modelSeed=%i_%i' % (seg, seed, split)
                        # if split == 0:
                        #     path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_0_noisesub=id_rand_10000_0_label_noise_0.2_%i_modelSeed=%i' % (seg, seed)
                        # else:
                        #     path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_0_noisesub=id_rand_10000_0_label_noise_0.2_%i_modelSeed=%i-%i' % (seg, seed, split)
                    else:
                        path = 'checkpoints/cifar100_sgd_mse_resnet18_width=%i_mom=0_9_sub=id_rand_10000_%i_modelSeed=%i_%i' % (width, seg, seed, split)

                if suffix:
                    path += suffix

                if config.phase.lower() == 'model':
                    state = 'model.pt'
                else:
                    state = 'model-%i.pt' % epoch
                net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel,
                              model=config.model, depth=config.depth, width=width, state=state,
                              device=config.device)
                net.eval()
                nets['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split] = net
    return nets


def get_risk_and_bias(outputs_all, y_true):
    risk_sum = 0
    outputs_sum = 0
    for seg in config.segs:
        for seed in config.seeds:
            for split in config.splits:
                outputs = outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
                risk_sum += np.sum((outputs - y_true)**2)
                outputs_sum += outputs
    risk_avg = risk_sum / len(config.segs) / len(config.seeds) / len(config.splits)
    outputs_avg = outputs_sum / len(config.segs) / len(config.seeds) / len(config.splits)
    bias = np.sum((outputs_avg - y_true) **2)
    return risk_avg, bias

def get_variance(outputs_all, y_true):
    outputs_list = []
    for seg in config.segs:
        for seed in config.seeds:
            for split in config.splits:
                outputs_list.append(outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split])
    outputs_avg = np.mean(outputs_list, axis=0)
    ve_all = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    outputs_list = []
    for seg in config.segs:
        outputs_list.append(np.mean([outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]\
                            for seed in config.seeds for split in config.splits], axis=0))
    ve_seg = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    outputs_list = []
    for seed in config.seeds:
        outputs_list.append(np.mean([outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]\
                            for seg in config.segs for split in config.splits], axis=0))
    ve_seed = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    outputs_list = []
    for split in config.splits:
        outputs_list.append(np.mean([outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]\
                            for seed in config.seeds for seg in config.segs], axis=0))
    ve_split = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    outputs_list = []
    for seg in config.segs:
        for seed in config.seeds:
            outputs_list.append(np.mean([outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]\
                                for split in config.splits], axis=0))
    ve_seg_seed = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    outputs_list = []
    for seg in config.segs:
        for split in config.splits:
            outputs_list.append(np.mean([outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]\
                                for seed in config.seeds], axis=0))
    ve_seg_split = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    outputs_list = []
    for seed in config.seeds:
        for split in config.splits:
            outputs_list.append(np.mean([outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]\
                                for seg in config.segs], axis=0))
    ve_seed_split = np.sum(np.mean((outputs_list - outputs_avg)**2, axis=0))

    var = defaultdict(None)
    var['seg'] = ve_seg
    var['seed'] = ve_seed
    var['split'] = ve_split
    var['seg-seed'] = ve_seg_seed - ve_seg - ve_seed
    var['seg-split'] = ve_seg_split - ve_seg - ve_split
    var['seed-split'] = ve_seed_split - ve_seed - ve_split
    var['seg-seed-split'] = ve_all - ve_seg_seed - ve_seg_split - ve_seed_split + ve_seg + ve_seed + ve_split
    return var

# def get_variance(outputs_all, y_true, level=None):
#     variance_sum = 0
#     if level == 'seed':
#         for seg in config.segs:
#             for split in config.splits:
#                 outputs_list = []
#                 for seed in config.seeds:
#                     outputs_list.append(outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split])
#                 outputs_avg = np.mean(outputs_list, axis=0)
#                 variance_sum += np.sum((outputs_list - outputs_avg) ** 2)
#         variance_avg = variance_sum / len(config.segs) / len(config.seeds) / len(config.splits)
#     elif level == 'split':
#         for seg in config.segs:
#             outputs_list = []
#             for split in config.splits:
#                 outputs_sum_inner = 0
#                 for seed in config.seeds:
#                     outputs_sum_inner += outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
#                 outputs_avg_inner = outputs_sum_inner / len(config.seeds)
#                 outputs_list.append(outputs_avg_inner)
#             outputs_avg = np.mean(outputs_list, axis=0)
#             variance_sum += np.sum((outputs_list - outputs_avg) ** 2)
#         variance_avg = variance_sum / len(config.segs) / len(config.splits)
#     elif level == 'seg':
#         outputs_list = []
#         for seg in config.segs:
#             outputs_sum_inner = 0
#             for seed in config.seeds:
#                 for split in config.splits:
#                     outputs_sum_inner += outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
#             outputs_avg_inner = outputs_sum_inner /len(config.seeds) / len(config.splits)
#             outputs_list.append(outputs_avg_inner)
#         outputs_avg = np.mean(outputs_list, axis=0)
#         variance_sum += np.sum((outputs_list - outputs_avg) ** 2)
#         variance_avg = variance_sum / len(config.segs)
#     else:
#         raise NotImplementedError(level)
#     return variance_avg

# def get_variance(outputs_all, y_true, level=None):
#     variance_sum = 0
#     if level == 'split':
#         for seg in config.segs:
#             for seed in config.seeds:
#                 outputs_list = []
#                 for split in config.splits:
#                     outputs_list.append(outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split])
#                 outputs_avg = np.mean(outputs_list, axis=0)
#                 variance_sum += np.sum((outputs_list - outputs_avg) ** 2)
#         variance_avg = variance_sum / len(config.segs) / len(config.seeds) / len(config.splits)
#     elif level == 'seed':
#         for seg in config.segs:
#             outputs_list = []
#             for seed in config.seeds:
#                 outputs_sum_inner = 0
#                 for split in config.splits:
#                     outputs_sum_inner += outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
#                 outputs_avg_inner = outputs_sum_inner / len(config.splits)
#                 outputs_list.append(outputs_avg_inner)
#             outputs_avg = np.mean(outputs_list, axis=0)
#             variance_sum += np.sum((outputs_list - outputs_avg) ** 2)
#         variance_avg = variance_sum / len(config.segs) / len(config.seeds)
#     elif level == 'seg':
#         outputs_list = []
#         for seg in config.segs:
#             outputs_sum_inner = 0
#             for seed in config.seeds:
#                 for split in config.splits:
#                     outputs_sum_inner += outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
#             outputs_avg_inner = outputs_sum_inner /len(config.seeds) / len(config.splits)
#             outputs_list.append(outputs_avg_inner)
#         outputs_avg = np.mean(outputs_list, axis=0)
#         variance_sum += np.sum((outputs_list - outputs_avg) ** 2)
#         variance_avg = variance_sum / len(config.segs)
#     else:
#         raise NotImplementedError(level)
#     return variance_avg


# def bias_variance_square_loss(outputs_all, y_true):
#     """
#         outputs_all: num_splits x num_examples x num_dimensions
#         y_true: num_examples
#     """
#     num_splits = len(outputs_all)
#     num_examples = outputs_all[0].shape[0]
#     num_dim = outputs_all[0].shape[1]
#     assert(len(y_true) == num_examples), (len(y_true), num_examples)
# 
#     # maybe don't use broadcast in case some dimensions are identical
#     avg_expected_loss = np.sum((outputs_all - y_true)**2) / num_examples / num_splits
# 
#     main_predictions = np.mean(outputs_all, axis=0)
#     avg_bias = np.sum((main_predictions - y_true)**2) / num_examples
#     avg_var = np.sum((main_predictions - outputs_all)**2) / num_examples / num_splits
# 
#     return avg_expected_loss, avg_bias, avg_var


def compute_bias_variance(config, nets, loaders):

    num_ex = 0
    risk = 0
    bias2 = 0
    variances = defaultdict(float)
    losses = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    acces = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    if config.loss_type == 'square':
        criterion = mse_one_hot(num_classes=loaders.num_classes)
    else:
        raise NotImplementedError(config.loss_type)

    for batch_idx, (inputs, targets, _) in enumerate(loaders.testloader):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        outputs_all = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
        for seg in config.segs:
            for seed in config.seeds:
                for split in config.splits:
                    net = nets['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split]
                    with torch.no_grad():
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        _, preds = outputs.max(1)
                        acc = preds.eq(targets).sum() / targets.size(0)
                    outputs_all['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split] = F.softmax(outputs, dim=1).cpu().numpy()
                    losses['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split] += loss.item() * inputs.size(0)
                    acces['seg=%i' % seg]['seed=%i' % seed]['split=%i' % split] += acc.item() * inputs.size(0)

        y_true = F.one_hot(targets, num_classes=outputs.size(1)).float().cpu().numpy()

        # risk_, bias_, var_ = bias_variance_square_loss(outputs_all, targets)
        risk_, bias2_ = get_risk_and_bias(outputs_all, y_true)
        risk += risk_
        bias2 += bias2_
        variance = get_variance(outputs_all, y_true)
        for level in powerset(config.variance_levels):
            # variances[level] += get_variance(outputs_all, y_true, level=level)
            variances[level] += variance[level]

        num_ex += inputs.size(0)

    risk /= num_ex
    bias2 /= num_ex
    for key in variances:
        variances[key] = variances[key] / num_ex
    loss_sum = 0
    for key1 in losses:
        for key2 in losses[key1]:
            for key3 in losses[key1][key2]:
                loss_sum += losses[key1][key2][key3] / num_ex
    loss_avg = loss_sum / len(config.segs) / len(config.seeds) / len(config.splits)
    acc_sum = 0
    for key1 in acces:
        for key2 in acces[key1]:
            for key3 in acces[key1][key2]:
                acc_sum += acces[key1][key2][key3] / num_ex
    acc_avg = acc_sum / len(config.segs) / len(config.seeds) / len(config.splits)
    # losses = [loss / num_ex for loss in losses]
    # acces = [acc / num_ex * 100 for acc in acces] # should acc be divided by num_splits? check previous result
    return risk, bias2, variances, loss_avg, acc_avg


if __name__ == '__main__':

    config = {
        'loss_type': 'square',
        'segs': [0, 1],
        'seeds': [7, 8, 9],
        'splits': [0, 1, 2],
        'kd': False,
        'variance_levels': ['seg', 'seed', 'split'],

        'phase': 'Epoch', # 'Model',
        # epoch-wise
        'epochs': range(0, 1000, 10),
        'width': 16,

        # model-wise
        'widths': [2, 4, 8, 10, 12, 16, 64],
        'epoch': 1000, 

        'model': 'resnet', # wrn
        'depth': 18,
        'resume': False, # True
        'gpu_id': 4,
        'dataset': 'cifar100',
        'data_dir': '/home/chengyu/Initialization/data',
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
    file_name += '_seg=%s' % '-'.join([str(seg) for seg in config.segs])
    file_name += '_seed=%s' % '-'.join([str(seed) for seed in config.seeds])
    file_name += '_split=%s' % '-'.join([str(split) for split in config.splits])
    if config.kd:
        file_name += '_kd'
    # file_name += '_seed_first'
    # file_name += '_modelseed'
    # file_name += '_symmetric'
    if config.phase.lower() == 'model':
        file_name += '_modelwise'
    elif config.phase.lower() == 'epoch':
        file_name += '_epochwise'
    else:
        raise KeyError(config.phase)

    save_dir = 'bias_variance'
    logger = Logger('tmp/%s/%s.txt' % (save_dir, file_name), title='log', resume=config.resume)
    base_names = [config.phase.capitalize()] 
    metrics = ['Risk', 'Bias']
    # metrics += ['Variance-%s' % level for level in config.variance_levels]
    metrics += ['Variance-%s' % level for level in powerset(config.variance_levels)]
    metrics += ['Loss', 'Acc']
    # metrics += ['Loss%i' % i for i in range(config.num_splits)]
    # metrics += ['Acc%i' % i for i in range(config.num_splits)]
    logger.set_names(base_names + metrics)

    if config.phase.lower() == 'epoch':
        # epoch-wise
        for epoch in config.epochs:
            nets = get_nets(config, loaders, epoch=epoch, width=config.width)
            risk, bias2, variances, loss, acc = compute_bias_variance(config, nets, loaders)
            str_cpl = '\n[%i] Risk: %.4f Bias: %.4f Variance: %.4f Loss: %.4f Acc: %.4f'
            print(str_cpl % (epoch, risk, bias2, variances['split'], loss, acc))
            logs = [epoch, risk, bias2]
            logs.extend([variances[level] for level in powerset(config.variance_levels)])
            logs.extend([loss, acc])
            logger.append(logs)

    elif config.phase.lower() == 'model':
        # model-wise 
        for width in config.widths:
            nets = get_nets(config, loaders, width=width, epoch=config.epoch)
            risk, bias2, variances, loss, acc = compute_bias_variance(config, nets, loaders)
            str_cpl = '\n[%i] Risk: %.4f Bias: %.4f Variance: %.4f Loss: %.4f Acc: %.4f'
            print(str_cpl % (width, risk, bias2, variances['split'], loss, acc))
            logs = [width, risk, bias2]
            logs.extend([variances[level] for level in powerset(config.variance_levels)])
            logs.extend([loss, acc])
            logger.append(logs)
    else:
        raise KeyError(config.phase)

    print('-- Finished.. %.3f mins' % ((time.time() - start) / 60.0))


