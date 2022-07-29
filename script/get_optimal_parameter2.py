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
from src.utils import Dict2Obj, Logger

"""
    The problem of two teachers:
        if with accuracy constraint:
        * The two teachers usually disagree a lot on an adversarial example, therefore they would rely mostly on the true label to increase accuracy
        if without accuracy constraint:
        * The robust teacher will dominate and it benefits the NLL much more
"""

def get_logits(net, loader, src_net=None, surro_net=None, clean_net=None, ad=False, config=None):
    # src_net only for generating adversarial examples
    if src_net is None:
        src_net = net

    # First: collect all the logits and labels for the validation set
    logits_list = []
    if surro_net is not None:
        logits_surro_list = []
    if clean_net is not None:
        logits_clean_list = []
    labels_list = []
    corrects = 0
    counts = 0
    for e, (inputs, labels, _) in enumerate(loader):
        print('[%i/%i]' % (e+1, len(loader)), end='\r')
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        if ad:
            inputs_, _ = attack(src_net, 
                                nn.CrossEntropyLoss().cuda(),
                                inputs, labels,
                                eps=config.eps,
                                pgd_alpha=config.pgd_alpha,
                                pgd_iter=config.pgd_iter,
                                adversary=config.adversary,
                                randomize=config.randomize,
                                is_clamp=True,
                                target=None, ## mark...
                                config=config)
        else:
            inputs_ = inputs
        with torch.no_grad():
            logits = net(inputs_)
            logits_list.append(logits)
            if surro_net is not None:
                logits_surro = surro_net(inputs_)
                logits_surro_list.append(logits_surro)
            if clean_net is not None:
                logits_clean = clean_net(inputs_)
                logits_clean_list.append(logits_clean)
        labels_list.append(labels)
        
        ## Check acc
        _, preds = logits.max(1)
        corrects += torch.sum(preds.eq(labels))
        counts += inputs.size(0)
        print('[%i/%i] acc: %.4f' % (e + 1, len(loader), corrects.float()/counts), end='\r')
    
        # to speed up, no need to check all examples
        ## but better to check all for validation loader
        if hasattr(config, 'max_iter') and config.max_iter and e > config.max_iter: 
            break

    logits_ = torch.cat(logits_list).cuda()
    if surro_net is not None:
        logits_surro = torch.cat(logits_surro_list).cuda()
    else:
        logits_surro = None
    if clean_net is not None:
        logits_clean = torch.cat(logits_clean_list).cuda()
    else:
        logits_clean = None
    labels_ = torch.cat(labels_list).cuda()
    return logits_, logits_surro, logits_clean, labels_


class ModelWithTemperature(nn.Module):
    """
        Instances for evluating the NLL loss on the validation set with different temperatures and interpolation ratios
    """
    def __init__(self, net, valid_loader, src_net=None, device=None, ad=False, surro_net=None, clean_net=None, config=None):
        super(ModelWithTemperature, self).__init__()
        self.net = net.to(device)
        if src_net is None:
            self.src_net = net
        else:
            self.src_net = src_net.to(device)
        self.surro_net = surro_net.to(device)
        self.clean_net = clean_net.to(device)
        self.config = config

        # store logits
        print('-- Get logits on validation set..')
        self.logits, self.logits_surro, self.logits_clean, self.labels = get_logits(self.net, valid_loader,
                                                                                    src_net=self.src_net,
                                                                                    surro_net=self.surro_net,
                                                                                    clean_net=self.clean_net,
                                                                                    ad=ad, config=self.config)
        
        # if self.surro_net is not None:
        #     print('-- Get logits on validation set for surrogate model..')
        #     # self.surro_logits, self.surro_labels = get_logits(self.surro_net, valid_loader, src_net=None, ad=ad, config=self.config)
        #     self.logits_surro, self.labels_surro = get_logits(self.surro_net, valid_loader, src_net=net, ad=ad, config=self.config)

        # if self.clean_net is not None:
        #     print('-- Get logits on validation set for clean model..')
        #     # self.surro_logits, self.surro_labels = get_logits(self.surro_net, valid_loader, src_net=None, ad=ad, config=self.config)
        #     self.logits_clean, self.labels_clean = get_logits(self.clean_net, valid_loader, src_net=net, ad=ad, config=self.config)

    def eval(self, temperature, coefficient=1, temperature_clean=2, coefficient_clean=0.2, verbose=True):
        nll_criterion = nn.NLLLoss().cuda() 
        temperature = torch.tensor([temperature]).cuda()
        nll = nll_criterion(self.get_log_probas(temperature, coefficient, temperature_clean, coefficient_clean), self.labels).item()
        if verbose:
            print('Temperature: %.3f Coefficient: %.3f NLL: %.3f' % (temperature.item(), coefficient, nll))
        return nll

    def get_log_probas(self, temperature, coefficient=1, temperature_clean=2, coefficient_clean=0.2):
        probas = F.softmax(self.logits / temperature, dim=1)
        probas_clean = F.softmax(self.logits_clean / temperature_clean, dim=1)
        probas = probas * coefficient + probas_clean * coefficient_clean
        if 1 - coefficient - coefficient_clean > 0:
            probas_surrogate = F.softmax(self.logits_surro / temperature, dim=1)
            probas += probas_surrogate * (1 - coefficient - coefficient_clean)
        return torch.log(probas + 1e-8)
    

class AccuracyWithTemperature(nn.Module):
    """
        Instances for evluating the accuracy of the alternative labeling with different temperatures and interpolation ratios
    """
    def __init__(self, net, train_loader, device=None, ad=False, clean_net=None, num_classes=10, config=None):
        super(AccuracyWithTemperature, self).__init__()
        self.net = net.to(device)
        self.clean_net = clean_net.to(device)
        self.num_classes = num_classes
        
        self.config = config
        
        # store logits
        print('-- Get logits on training set..')
        # self.logits, self.labels = get_logits(self.net, train_loader, ad=ad, config=self.config)
        self.logits, self.logits_surro, self.logits_clean, self.labels = get_logits(self.net, train_loader,
                                                                                    src_net=self.net,
                                                                                    surro_net=None,
                                                                                    clean_net=self.clean_net,
                                                                                    ad=ad, config=self.config)

        # if self.clean_net is not None:
        #     print('-- Get logits on training set for clean model..')
        #     self.logits_clean, self.labels_clean = get_logits(self.clean_net, train_loader, src_net=net, ad=ad, config=self.config)

    def eval_accuracy(self, temperature=2, coefficient=1, temperature_clean=2, coefficient_clean=0.2):
        probas = self.get_kd_probas(temperature=temperature, coefficient=coefficient, temperature_clean=temperature_clean, coefficient_clean=coefficient_clean)
        _, preds = probas.max(1)
        return (preds.eq(self.labels).sum() / self.labels.size(0)).item()      
    
    def get_kd_probas(self, temperature=2, coefficient=1, temperature_clean=2, coefficient_clean=0.2):
        probas = F.softmax(self.logits / temperature, dim=1)
        probas_clean = F.softmax(self.logits_clean / temperature_clean, dim=1)
        probas = probas * coefficient + probas_clean * coefficient_clean
        if 1 - coefficient - coefficient_clean > 0:
            labels_one_hot = F.one_hot(self.labels, num_classes=self.num_classes).float()
            probas += labels_one_hot * (1 - coefficient - coefficient_clean)
        return probas


def main():
    # -- hypers
    # path = 'checkpoints/svhn_sgd_PreActResNet18_ad_pgd_10_alpha=1_cosine_lr=0.01_wd=0_0005_mom=0_9_pgd_10_nval=7325'
    path = 'checkpoints/svhn_sgd_PreActResNet18_ad_pgd_10_alpha=1_lr=1e-02_wd=0_0005_mom=0_9_pgd_10_nval=7325-1'
    path_clean = 'checkpoints/svhn_sgd_PreActResNet18_lr=1e-02_wd=0_0005_mom=0_9_nval=7325-1'
    dataset = 'svhn' # 'cifar10'
    model = 'PreActResNet18'
    depth = 18
    width = 10
    state = 'best_model_val'
    state_surro = 'model'
    max_iter = 5 # None
    gpu_id = '5'
    valset = 'valloader'

    acc_min = None # 0.9 # Minimum accuracy of the alternative labeling. if none, no constraint on the accuracy
    ad = True # If evaluate on adversarial examples
    interpolation = True # If interpolating probability with the given labels

    # -- env
    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # config
    config = {'dataset': dataset,
              'soft_label': False,
              'soft_label_test': False,
              'valsize': 7325,
              'shuffle_train_loader': False,
              'random_augment': False,
              'adversary': 'pgd', # if evaluate on adversarial examples, we use PGD attack
              'eps': 8, 
              'pgd_alpha': 2,
              'pgd_iter': 10,
              'randomize': True,
              'device': device,
              'max_iter': max_iter,
             }
    config = Dict2Obj(config)

    # scale epsilon
    config.eps = scale_step(config.eps, dataset=config.dataset, device=config.device)
    config.pgd_alpha = scale_step(config.pgd_alpha, dataset=config.dataset, device=config.device)

    # -- get loader
    print('>>>>>>>>>>> get loader..')
    loaders = get_loaders(dataset=dataset,
                          data_dir='/home/chengyu/Initialization/data',
                          shuffle_train_loader=config.shuffle_train_loader, 
                          config=config)
    valloader = getattr(loaders, valset)
    trainloader = getattr(loaders, 'trainloader')

    # -- get model
    print('>>>>>>>>>>> get net..')
    if state == 'last':
        model_state = 'model.pt'
    elif state == 'best':
        model_state = 'best_model.pt'
    elif str(state).isdigit(): 
        model_state = 'model-%i.pt' % int(state)
    elif any(char.isdigit() for char in state):
        phase, idx = state.split('-')
        model_state = '%s_model-%s.pt' % (phase, idx)
    else:
        model_state = '%s.pt' % state
        # raise KeyError(state)
    print(model_state)
    net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state=model_state, device=device)
    clean_net = get_net(path_clean, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state=model_state, device=device)
    if interpolation:
        model_state_surro = '%s.pt' % state_surro
        print(model_state_surro)
        surro_net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state=model_state_surro, device=device)
    else:
        surro_net = None

    # -- Grid search - 2D
    print('>>>>>>>>>>> Initialize evaluation instance..')
    scaled_model = ModelWithTemperature(net, valloader, surro_net=surro_net, clean_net=clean_net, device=device, ad=ad, config=config)
    acc_model = AccuracyWithTemperature(net, trainloader, clean_net=clean_net, num_classes=loaders.num_classes, device=device, ad=ad, config=config)

    print('>>>>>>>>>>> Start grid searching for optimal hyperparameters..')
    temperatures = np.exp(np.linspace(np.log(0.2), np.log(16), 10))
    temperatures_clean = np.exp(np.linspace(np.log(0.2), np.log(16), 10))
    coefficients = np.linspace(0.01, 1.0, 10)
    coefficients_clean = np.linspace(0.01, 1.0, 10)
    nlls = np.zeros((len(temperatures), len(coefficients), len(temperatures_clean), len(coefficients_clean)))
    accs = np.zeros((len(temperatures), len(coefficients), len(temperatures_clean), len(coefficients_clean)))

    for i, temperature in enumerate(temperatures):
        for j, coefficient in enumerate(coefficients):
            for i, temperature_clean in enumerate(temperatures_clean):
                for j, coefficient_clean in enumerate(coefficients_clean):
                    print('T = %g, rho = %g, T_st = %g, rho_st = %g' % (temperature, coefficient, temperature_clean, coefficient_clean), end='\r')
                    nlls[i, j] = scaled_model.eval(temperature=temperature,
                                                   coefficient=coefficient,
                                                   temperature_clean=temperature_clean,
                                                   coefficient_clean=coefficient_clean,
                                                   verbose=False)

                    accs[i, j] = acc_model.eval_accuracy(temperature=temperature,
                                                         coefficient=coefficient,
                                                         temperature_clean=temperature_clean,
                                                         coefficient_clean=coefficient_clean,
                                                         )

    if acc_min is None:
        idx_min = np.unravel_index(np.argmin(nlls), nlls.shape)
        temperature_optimal = temperatures[idx_min[0]]
        coefficient_optimal = coefficients[idx_min[1]]
        temperature_clean_optimal = temperatures_clean[idx_min[2]]
        coefficient_clean_optimal = coefficients_clean[idx_min[3]]
        nll_optimal = nlls[idx_min]
        acc_optimal = accs[idx_min]
    else:
        nlls_masked = np.array(nlls)
        nlls_masked[accs < acc_min] = np.max(nlls_masked) * 2
        idx_min = np.unravel_index(np.argmin(nlls_masked), nlls_masked.shape)
        temperature_optimal = temperatures[idx_min[0]]
        coefficient_optimal = coefficients[idx_min[1]]
        temperature_clean_optimal = temperatures_clean[idx_min[2]]
        coefficient_clean_optimal = coefficients_clean[idx_min[3]]
        nll_optimal = nlls_masked[idx_min]
        acc_optimal = accs[idx_min]

    print('T* = %.2f, rho* = %.2f, T_st* = %.2f, rho_st* = %.2f, nll_min = %.2f, acc = %.2f' % (temperature_optimal, coefficient_optimal, temperature_clean_optimal, coefficient_clean_optimal, nll_optimal, acc_optimal))
    logger = Logger('%s/log_optimal_parameter_%s.txt' % (path, state))
    logger.set_names(['T*', 'rho*', 'T_st*', 'rho_st*', 'nll_min', 'acc'])
    logger.append([temperature_optimal, coefficient_optimal, temperature_clean_optimal, coefficient_clean_optimal, nll_optimal, acc_optimal])
    logger.close()

    print('>>>>>>>>>>> save results..')
    save_data = {'temperature': temperatures,
                 'ratio': coefficients,
                 'temperature_clean': temperatures_clean,
                 'ratio_clean': coefficients_clean,
                 'nll': nlls,
                 'accs': accs}
    torch.save(save_data, '%s/log_optimal_parameter_%s.pt' % (path, state))


if __name__ == '__main__':

    main()
