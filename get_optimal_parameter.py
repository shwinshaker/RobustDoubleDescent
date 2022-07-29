#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

from src.adversary import attack, scale_step
from src.preprocess import get_loaders
from src.analyses import get_net
from src.utils import Dict2Obj

class ModelWithTemperature(nn.Module):
    """
        Instances for evluating the NLL loss on the validation set with different temperatures and interpolation ratios
    """
    def __init__(self, net, valid_loader, src_net=None, device=None, ad=True, surro_net=None):
        super(ModelWithTemperature, self).__init__()
        self.net = net.to(device)
        self.ad = ad
        if src_net is None:
            self.src_net = net
        else:
            self.src_net = src_net
        self.surro_net = surro_net
        
        # config
        config = {'dataset': 'cifar10',
                  'device': device,
                  'eps': 8, 
                  'pgd_alpha': 2,
                  'pgd_iter': 10,
                  'adversary': 'pgd',
                  'randomize': True,
                 }
        self.config = Dict2Obj(config)
        
        # scale epsilon
        self.config.eps = scale_step(self.config.eps,
                                     dataset=self.config.dataset,
                                     device=self.config.device)
        self.config.pgd_alpha = scale_step(self.config.pgd_alpha,
                                           dataset=self.config.dataset,
                                           device=self.config.device)
        
        # store logits
        self.logits, self.labels = self.get_logits(self.net, self.src_net, valid_loader)
        
        if self.surro_net is not None:
            self.surro_logits, self.surro_labels = self.get_logits(self.surro_net, self.surro_net, valid_loader)

    def eval_nll(self, temperature, coefficient=0, verbose=True):
        nll_criterion = nn.NLLLoss().cuda() 
        temperature = torch.tensor([temperature]).cuda()
        nll = nll_criterion(self.get_log_probas(temperature, coefficient), self.labels).item()
        if verbose:
            print('Temperature: %.3f Coefficient: %.3f NLL: %.3f' % (temperature.item(), coefficient, nll))
        return nll

    def get_log_probas(self, temperature, coefficient=0):
        probas = F.softmax(self.logits / temperature, dim=1)
        if coefficient > 0:
            probas *= coefficient
            probas_surrogate = F.softmax(self.surro_logits / temperature, dim=1)
            probas += probas_surrogate * (1 - coefficient)
        return torch.log(probas + 1e-8)
    
    def get_logits(self, net, src_net, valid_loader):
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        for e, (inputs, labels, _) in enumerate(valid_loader):
            print('[%i/%i]' % (e+1, len(valid_loader)), end='\r')
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
            if self.ad:
                inputs_, _ = attack(src_net, 
                                   nn.CrossEntropyLoss().cuda(),
                                   inputs, labels,
                                   eps=self.config.eps,
                                   pgd_alpha=self.config.pgd_alpha,
                                   pgd_iter=self.config.pgd_iter,
                                   adversary=self.config.adversary,
                                   randomize=self.config.randomize,
                                   is_clamp=True,
                                   target=None, ## mark...
                                   config=self.config)
            else:
                inputs_ = inputs
            with torch.no_grad():
                logits = net(inputs_)
            logits_list.append(logits)
            labels_list.append(labels)
            
        logits_ = torch.cat(logits_list).cuda()
        labels_ = torch.cat(labels_list).cuda()
        return logits_, labels_


class AccuracyWithTemperature(nn.Module):
    """
        Instances for evluating the accuracy of the alternative labeling with different temperatures and interpolation ratios
    """
    def __init__(self, net, train_loader, device=None, ad=True, num_classes=10):
        super(AccuracyWithTemperature, self).__init__()
        self.net = net.to(device)
        self.ad = ad
        self.num_classes = num_classes
        
        # config
        config = {'dataset': 'cifar10',
                  'device': device,
                  'eps': 8, 
                  'pgd_alpha': 2,
                  'pgd_iter': 10,
                  'adversary': 'pgd',
                  'randomize': True,
                 }
        self.config = Dict2Obj(config)
        
        # scale epsilon
        self.config.eps = scale_step(self.config.eps,
                                     dataset=self.config.dataset,
                                     device=self.config.device)
        self.config.pgd_alpha = scale_step(self.config.pgd_alpha,
                                           dataset=self.config.dataset,
                                           device=self.config.device)
        
        # store logits
        self.logits, self.labels = self.get_logits(self.net, train_loader)

    def eval_accuracy(self, temperature=2, coefficient=0):
        probas = self.get_kd_probas(temperature=temperature, coefficient=coefficient)
        _, preds = probas.max(1)
        return (preds.eq(self.labels).sum() / self.labels.size(0)).item()      
    
    def get_kd_probas(self, temperature=2, coefficient=0):
        probas = F.softmax(self.logits / temperature, dim=1)
        labels_one_hot = F.one_hot(self.labels, num_classes=self.num_classes).float()
        probas *= coefficient
        probas += labels_one_hot * (1 - coefficient)
        return probas

    def get_logits(self, net, loader):
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        corrects = 0
        counts = 0
        for e, (inputs, labels, _) in enumerate(loader):
            print('[%i/%i]' % (e, len(loader)), end='\r')
            inputs, labels = inputs.to(device), labels.to(device)
            if self.ad:
                inputs_, _ = attack(net, 
                                    nn.CrossEntropyLoss().cuda(),
                                    inputs, labels,
                                    eps=self.config.eps,
                                    pgd_alpha=self.config.pgd_alpha,
                                    pgd_iter=self.config.pgd_iter,
                                    adversary=self.config.adversary,
                                    randomize=self.config.randomize,
                                    is_clamp=True,
                                    target=None, ## mark...
                                    config=self.config)
            else:
                inputs_ = inputs
            with torch.no_grad():
                logits = net(inputs_)
            logits_list.append(logits)
            labels_list.append(labels)
            
            ## Check acc
            _, preds = logits.max(1)
            corrects += torch.sum(preds.eq(labels))
            counts += inputs.size(0)
            print('[%i/%i] acc: %.4f' % (e + 1, len(loader), corrects.float()/counts), end='\r')
        
            ## to speed up, no need to check all examples
            # if e > 100: 
            #     break
            
        logits_ = torch.cat(logits_list).cuda()
        labels_ = torch.cat(labels_list).cuda()
        return logits_, labels_


def main():
    # -- hypers
    dataset = 'cifar10'
    path = 'checkpoints/sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
    model = 'PreActResNet18'
    depth = 28
    width = 5
    gpu_id = '0'
    acc_min = 0.9 # Minimum accuracy of the alternative labeling. if none, no constraint on the accuracy

    # -- env
    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- get loader
    print('>>>>>>>>>>> get loader..')
    loaders = get_loaders(dataset=dataset,
                          random_augment=False,
                          shuffle_train_loader=False, 
                          data_dir='/home/chengyu/Initialization/data')

    # -- get model
    print('>>>>>>>>>>> get net..')
    net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state='best_model.pt', device=device)
    net_last = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state='model.pt', device=device)

    # -- Grid search - 2D
    print('>>>>>>>>>>> Initialize evaluation instance..')
    scaled_model = ModelWithTemperature(net, loaders.testloader, surro_net=net_last, device=device)
    acc_model = AccuracyWithTemperature(net, loaders.trainloader, num_classes=loaders.num_classes, device=device)

    print('>>>>>>>>>>> Start grid searching for optimal hyperparameters..')
    temperatures = np.exp(np.linspace(np.log(0.2), np.log(16), 100))
    coefficients = np.linspace(0.01, 1.0, 100)
    nlls = np.zeros((len(temperatures), len(coefficients)))
    accs = np.zeros((len(temperatures), len(coefficients)))

    for i, temperature in enumerate(temperatures):
        for j, coefficient in enumerate(coefficients):
            print('T = %g, rho = %g' % (temperature, coefficient), end='\r')
            nlls[i, j] = scaled_model.eval_nll(temperature=temperature,
                                               coefficient=coefficient,
                                               verbose=False)

            accs[i, j] = acc_model.eval_accuracy(temperature=temperature,
                                                 coefficient=coefficient)

    if acc_min is None:
        idx_min = np.unravel_index(np.argmin(nlls), nlls.shape)
        print('T* = %.2f, rho* = %.2f, nll_min = %.2f' % (temperatures[idx_min[0]], coefficients[idx_min[1]], nlls[idx_min]))
    else:
        nlls_masked = np.array(nlls)
        nlls_masked[accs < acc_min] = np.max(nlls_masked) * 2
        idx_min = np.unravel_index(np.argmin(nlls_masked), nlls_masked.shape)
        print('T* = %.2f, rho* = %.2f, nll_min = %.2f' % (temperatures[idx_min[0]], coefficients[idx_min[1]], nlls_masked[idx_min]))

    print('>>>>>>>>>>> save results..')
    save_data = {'temperature': temperatures,
                 'ratio': coefficients,
                 'nll': nlls,
                 'accs': accs}
    torch.save(save_data, '%s/log_optimal_parameter.pt' % path)


if __name__ == '__main__':

    main()
