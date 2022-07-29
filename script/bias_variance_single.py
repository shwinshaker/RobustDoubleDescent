#!./env python
import torch
import numpy as np
import torch.nn.functional as F

import os
import time
from collections.abc import Iterable

from src.utils import Logger
from src.preprocess import get_loaders, get_loaders_augment
from src.analyses import get_ad_examples, get_net
from src.utils import Dict2Obj
from src.utils import mse_one_hot

def get_nets(config, epoch, loaders):
    nets = []
    for i_seed, seed in enumerate(config.seed):
        nets.append([])
        for split in range(config.num_splits):
            # print('Model - %i' % split)
            # if eps > 0 or epst > 0:
            #     ad_name = '_pgd'
            # else:
            #     ad_name = ''

            # if friendly == 50000:
            #     str_friend = ''
            # else:
            #     str_friend = '_mindist_friend_%i_balance' % friendly

            # if split == 0:
            #     subset_path = 'mean_rb_id_pgd10%s_rand_5000' % str_friend
            #     # subset_path = 'mean_rb_id_pgd10_mindist_friend_5000_10th_balance'
            # else:
            #     subset_path = 'mean_rb_id_pgd10%s_rand_5000_%i' % (str_friend, split)
            #     # asubset_path = 'mean_rb_id_pgd10_mindist_friend_5000_10th_balance-%i' % split

            # path = 'checkpoints/adam_wrn-28-%i_gain=1_0_ad_pgd_10_eps=%i_alpha=1_lr=1e-04_mom=0_9_pgd_10_epst=%i_sub=%s' % (width, eps, epst, subset_path)
            # path = 'checkpoints/ad_aug%s_eps=%i_epst=%i_adam_wrn-28-%i_gain=1_0_lr=1e-04_mom=0_9_sub=%s' % (ad_name, eps, epst, width, subset_path)

            if config.kd:
                path = 'checkpoints/sgd_mse_resnet18_kd_T=2_st=0.5_mom=0_9_sub=id_rand_10000_0_noisesub=id_rand_10000_0_label_noise_0.2_%i_seed=%i' % (split, seed)
            else:
                path = 'checkpoints/sgd_mse_resnet18_mom=0_9_sub=id_rand_10000_0_noisesub=id_rand_10000_0_label_noise_0.2_%i_seed=%i' % (split, seed)

            net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel,
                          model=config.model, depth=config.depth, width=config.width, state='model-%i.pt' % epoch,
                          device=config.device)
            net.eval()
            nets[-1].append(net)
    if len(config.seed) == 1:
        nets = nets[0]
    return nets


def wasserstein(outputs1, outputs2):
    return (outputs1 * outputs2).mean(dim=1).sum()


def bias_variance_square_loss(outputs_all, y_true):
    """
        outputs_all: num_splits x num_examples x num_dimensions
        y_true: num_examples
    """
    num_splits = len(outputs_all)
    num_examples = outputs_all[0].shape[0]
    num_dim = outputs_all[0].shape[1]
    assert(len(y_true) == num_examples), (len(y_true), num_examples)

    outputs_all = np.array([F.softmax(outputs, dim=1).cpu().numpy() for outputs in outputs_all])
    y_true = F.one_hot(y_true, num_classes=num_dim).float().cpu().numpy()

    # maybe don't use broadcast in case some dimensions are identical
    avg_expected_loss = np.sum((outputs_all - y_true)**2) / num_examples / num_splits

    main_predictions = np.mean(outputs_all, axis=0)
    avg_bias = np.sum((main_predictions - y_true)**2) / num_examples
    avg_var = np.sum((main_predictions - outputs_all)**2) / num_examples / num_splits

    return avg_expected_loss, avg_bias, avg_var


def bias_variance_0_1_loss(outputs_all, y_true):
    """
        bias-variance decomposition of 0-1 loss - `A Unified Bias-Variance Decomposition for Zero-One and Squared Loss`
        'https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bias_variance_decomp.py#L129'
        Parameters:
            * all_pred: num_splits x num_examples
            * y_true: num_examples
        Problems:
            * In this theory, if variance can reduce risk (when main prediction is not correct),
                seems not appropriate to say variance causes the overfitting? (Risk increase).
                At least we can say variance definitely causes overfitting,
                because since the bias doesn't change a lot, variance should be the only source
    """
    all_pred = [outputs.max(1).cpu().numpy() for outputs in outputs_all]
    y_true = y_true.cpu().numpy()

    avg_expected_loss = np.apply_along_axis(lambda x:
                                            (x != y_true).mean(),
                                            axis=1,
                                            arr=all_pred).mean()

    main_predictions = np.apply_along_axis(lambda x:
                                           np.argmax(np.bincount(x)),
                                           axis=0,
                                           arr=all_pred)
    # print(main_predictions)

    avg_bias = np.sum(main_predictions != y_true) / y_true.size

    var = np.zeros(all_pred[0].shape)
    for pred in all_pred:
        var += (pred != main_predictions).astype(np.int)
    var /= len(all_pred)

    avg_var = var.sum() / y_true.shape[0]

    return avg_expected_loss, avg_bias, avg_var


def compute_bias_variance(config, loaders, epoch=10):
    nets = get_nets(config, epoch, loaders)

    risk = 0
    bias2 = 0
    variance = 0
    num_ex = 0
    losses = [0] * config.num_splits
    acces = [0] * config.num_splits

    if config.loss_type == 'square':
        criterion = mse_one_hot(num_classes=loaders.num_classes)
    elif config.loss_type in ['0-1', 'ce']:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(config.loss_type)

    for batch_idx, (inputs, targets, _) in enumerate(loaders.testloader):
        # print('[%i / %i]' % (batch_idx + 1, len(loaders.testloader)), end='\r')
        inputs_, targets = inputs.to(config.device), targets.to(config.device)

        outputs_all = []
        # preds_all = []
        for en, nets_ in enumerate(nets):
            for net in nets_:
                ## But the adversary will also bring difference here -> Variance?
                if config.adversary:
                    inputs, _ = get_ad_examples(net, inputs_, labels=targets,
                                                adversary='pgd', eps=config.epst, pgd_alpha=2, pgd_iter=10,
                                                dataset='cifar10', device=config.device)
                else:
                    inputs = inputs_
                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    _, preds = outputs.max(1)
                    acc = preds.eq(targets).sum()
                    # outputs = F.softmax(outputs, dim=1)
            losses[en] += loss.item() * inputs.size(0)
            acces[en] += acc.item()
            outputs_all.append(outputs)
            # preds_all.append(preds.cpu().numpy())

        if config.loss_type == 'square':
            risk_, bias_, var_ = bias_variance_square_loss(outputs_all, targets)
        elif config.loss_type == '0-1':
            # risk_, bias_, var_ = bias_variance_0_1_loss(preds_all, targets.cpu().numpy())
            risk_, bias_, var_ = bias_variance_0_1_loss(outputs_all, targets)
        risk += risk_ * inputs.size(0)
        bias2 += bias_ * inputs.size(0)
        variance += var_ * inputs.size(0)
        num_ex += inputs.size(0)

        # with torch.no_grad():
        #     # average and normalize
        #     outputs_avg = torch.stack(outputs_all).log().mean(dim=0)
        #     outputs_avg = outputs_avg.exp()
        #     outputs_avg = outputs_avg / outputs_avg.sum(dim=1, keepdims=True)

        #     # bias is the cross entropy loss of averaged outputs
        #     # bias2 += F.nll_loss(outputs_avg.log(), targets, reduction='sum')
        #     # bias2 += wasserstein(outputs_avg, targets)
        #     # 0-1 loss # true class only
        #     _, preds_avg = outputs_avg.max(1)
        #     bias2 += preds_avg.eq(targets).sum().float()
        #     # hamming loss # track all classes

        #     # variance is the variance of the outputs of different models
        #     for outputs in outputs_all:
        #         # variance += F.kl_div(outputs.log(), outputs_avg, reduction='none').sum()
        #         # variance += wasserstein(outputs, outputs_avg)
        #         _, preds = outputs.max(1)
        #         variance += preds_avg.eq(preds).sum().float()

    losses = [loss / num_ex for loss in losses]
    acces = [acc / num_ex * 100 for acc in acces] # should acc be divided by num_splits? check previous result
    return risk / num_ex, bias2 / num_ex, variance / num_ex, losses, acces


if __name__ == '__main__':

    config = {
        'loss_type': 'square',
        'seg': [0],
        'seed': [7, 8, 9],
        'split': [0, 1, 2],
        'kd': True,

        'model': 'resnet', # wrn
        'depth': 18,
        'width': 16, # 5
        'resume': False, # True
        'gpu_id': 2,
        'dataset': 'cifar10',
        'data_dir': '/home/chengyu/Initialization/data',
        'epoch_start': 0,
        'epoch_end': 1000,

        'adversary': False,
        'eps': 4,
        'epst': 4,

        'aug': False, # mode= 'ad' # ad_aug
        'friendly': 50000,
    }
    config = Dict2Obj(config)
    if not isinstance(config.seed, Iterable):
        config.seed = [config.seed]

    ## Set device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(config.device)
    
    start = time.time()

    ## get loader
    if not config.aug:
        loaders = get_loaders(dataset=config.dataset,
                              random_augment=False,
                              shuffle_train_loader=False,
                              data_dir=config.data_dir)
    else:
        raise NotImplementedError('loader function changed, change accordingly')
        loaders = get_loaders_ad_augment(dataset=config.dataset,
                                         ad_augment='pgd',
                                         aug_eps_train=config.eps,
                                         aug_eps_test=config.epst,
                                         random_augment=False,
                                         shuffle_train_loader=False,
                                         data_dir=config.data_dir)

    ### computer bias variance
    # file_name = 'log_bias_variance_ad_wrn-%i' % width
    file_name = 'log_bias_variance_%s_resnet-%i-%i' % (config.loss_type, config.depth, config.width)
    if config.num_splits != 2:
        file_name += '_split=%i' % config.num_splits
    if len(config.seed) == 1:
        file_name += '_seed=%i' % config.seed[0]
    if config.kd:
        file_name += '_kd'
    if config.friendly == 50000:
        str_friend = ''
    else:
        str_friend = '_friend_%i' % config.friendly
    if config.adversary:
        file_name += '_eps=%i_epst=%i%s' % (config.eps, config.epst, str_friend)
        # file_name += '_eps=%i_epst=%i_segment-10th' % (eps, epst)

    save_dir = 'bias_variance'
    if config.aug:
        save_dir = 'aug_' + save_dir
    if config.adversary:
        save_dir = 'ad_' + save_dir
    logger = Logger('tmp/%s/%s.txt' % (save_dir, file_name), title='log', resume=config.resume)
    base_names = ['Epoch']
    metrics = ['Risk', 'Bias', 'Variance']
    metrics += ['Loss%i' % i for i in range(config.num_splits)]
    metrics += ['Acc%i' % i for i in range(config.num_splits)]
    logger.set_names(base_names + metrics)

    risks = []
    bias2s = []
    variances = []
    for epoch in range(config.epoch_start, config.epoch_end, 10):
        risk, bias2, variance, losses, acces = compute_bias_variance(config, loaders, epoch=epoch)
                                                                     #, ad=adversary, device=device, 
                                                                     # model=model, depth=depth, width=width, num_splits=num_splits,
                                                                     # eps=eps, epst=epst, friendly=friendly, loss_type=loss_type, seed=seed)
        risks.append(risk)
        bias2s.append(bias2)
        variances.append(variance)
        str_cpl = '\n[%i] Risk: %.4f Bias: %.4f Variance: %.4f Loss1: %.4f Loss2: %.4f Acc1: %.4f Acc2: %.4f'
        print(str_cpl % (epoch, risk, bias2, variance, losses[0], losses[1], acces[0], acces[1]))
        logs = [epoch, risk, bias2, variance]
        logs.extend(losses)
        logs.extend(acces)
        logger.append(logs)

    print('-- Finished.. %.3f mins' % ((time.time() - start) / 60.0))


