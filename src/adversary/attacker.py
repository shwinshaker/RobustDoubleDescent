#!./env python
# https://adversarial-ml-tutorial.org/adversarial_training/

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import accuracy, AverageMeter, GroupMeter, alignment, criterion_r
from ..preprocess import dataset_stats
from ..utils import DeNormalizer # used for autoattack training only
from .aa_attacker import AAAttacker # used for autoattack training only

__all__ = ['attack', 'ad_test', 'scale_step']


def scale_step(v, dataset, device='cpu'):
    # scale the epsilon based on stats in each channel
    n_channel = len(dataset_stats[dataset]['std'])
    std = torch.tensor(dataset_stats[dataset]['std']).view(n_channel, 1, 1).to(device)
    return v / 255. / std

def get_range(mean, std):
    n_channel = len(mean)
    mean_ = torch.tensor(mean).view(n_channel, 1, 1)
    std_ = torch.tensor(std).view(n_channel, 1, 1)
    return {'upper': (1 - mean_) / std_, 
            'lower': (0 - mean_) / std_}

dataset_range = dict([(dataset, get_range(dataset_stats[dataset]['mean'],
                                          dataset_stats[dataset]['std'])) for dataset in dataset_stats])

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def rand_sphere(eps, size, device=None, requires_grad=True):
    delta = torch.rand(size, requires_grad=requires_grad, device=device) # range = (0, 1] 
    delta.data = delta.data * 2 * eps - eps # make linf norm = eps
    return delta

def fgsm(net, criterion, X, y, eps=0.1, rand_init=False, config=None):
    """ 
        Generate FGSM adversarial examples on the examples X
            # fgsm (single step)
            # pgd (multiple step)
            # CW (optimize difference between correct and incorrect logits)
    """
    # aligned with FAST
    # net.train() 

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    loss = criterion(net(X + delta), y)

    loss.backward()
    delta.data = delta + eps * delta.grad.detach().sign()
    delta.data = clamp(delta, -eps, eps)
    delta.data = clamp(delta, delta_lower, delta_upper)

    # net.eval()

    return clamp(X + delta.detach(),
                 dataset_range[config.dataset]['lower'].to(config.device),
                 dataset_range[config.dataset]['upper'].to(config.device))


def pgd_linf(net, criterion, X, y, eps=0.1, alpha=0.02, num_iter=5, rand_init=False, config=None):

    delta_lower = dataset_range[config.dataset]['lower'].to(config.device) - X
    delta_upper = dataset_range[config.dataset]['upper'].to(config.device) - X

    if rand_init:
        delta = rand_sphere(eps, X.size(), device=config.device, requires_grad=True)
        delta.data = clamp(delta, delta_lower, delta_upper)
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = criterion(net(X + delta), y)

        loss.backward()
        delta.data = delta + alpha * delta.grad.detach().sign()
        delta.data = clamp(delta, -eps, eps)
        delta.data = clamp(delta, delta_lower, delta_upper)
        delta.grad.zero_()

    return clamp(X + delta.detach(),
                 dataset_range[config.dataset]['lower'].to(config.device),
                 dataset_range[config.dataset]['upper'].to(config.device))


def attack(net, criterion, X, y, weight=None, adversary='fgsm', eps=0.1, pgd_alpha=0.02, pgd_iter=5, norm='linf', target=None, get_steps=False, get_minimum=False, randomize=False, config=None):
    # TODO: remove weight, target, get_steps, get_minium, is_clamp and associated external variables

    if adversary == 'fgsm':
        return fgsm(net, criterion, X, y, eps=eps, rand_init=randomize, config=config)
    elif adversary == 'pgd':
        return pgd_linf(net, criterion, X, y, eps=eps, alpha=pgd_alpha, num_iter=pgd_iter, rand_init=randomize, config=config)
    elif adversary.lower() == 'aa':
        # While using autoattack, 'eps' is neglected as it is scaled, instead used the value in config
        denormalize = DeNormalizer(dataset_stats[config.dataset]['mean'],
                                   dataset_stats[config.dataset]['std'],
                                   X.size(1), config.device)
        attacker = AAAttacker(net=net,
                              eps=config.eps_, # use unscaled eps
                              normalize=True,
                              mode='fast',
                              path='.',
                              device=config.device,
                              data_dir=None)
        X_ = denormalize(X)
        X_ad, _ = attacker.evaluate(x_test=X_, y_test=y)
        X_ad = attacker._normalize(X_ad)
        return X_ad
    else:
        raise KeyError(adversary)


def ad_test(dataloader, net, criterion, config, classes=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    if hasattr(config, 'class_eval') and config.class_eval:
        top1_class = GroupMeter(classes)

    for i, tup in enumerate(dataloader, 0):
        if len(tup) == 2:
            inputs, labels = tup
        else:
            inputs, labels, _ = tup
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        inputs_ad = attack(net, criterion, inputs, labels,
                           adversary=config.ad_test, eps=config.eps_test, pgd_alpha=config.pgd_alpha_test, pgd_iter=config.pgd_iter_test,
                           config=config)  

        with torch.no_grad():
            outputs_ad = net(inputs_ad)
            loss_ad = criterion(outputs_ad, labels)
        prec_ad, = accuracy(outputs_ad.data, labels.data)

        losses.update(loss_ad.item(), inputs.size(0))        
        top1.update(prec_ad.item(), inputs.size(0))

        if hasattr(config, 'class_eval') and config.class_eval:
            top1_class.update(outputs_ad, labels)

    extra_metrics = dict()
    if hasattr(config, 'class_eval') and config.class_eval:
        extra_metrics['class_acc'] = top1_class.output_group()
    
    return losses.avg, top1.avg, extra_metrics # robust_score_avg, top1_class

