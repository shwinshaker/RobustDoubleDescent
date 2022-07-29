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
from src.utils import Confidence

"""
    Why teachers are not evaluated on the same set of adversarial examples?
        * Surrogate model?
        * Standard teacher?
"""

def get_logits(net, loader, src_net=None, ad=False, config=None):
    # src_net only for generating adversarial examples
    if src_net is None:
        src_net = net

    # First: collect all the logits and labels for the validation set
    logits_list = []
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
    labels_ = torch.cat(labels_list).cuda()
    return logits_, labels_


class ModelWithTemperature(nn.Module):
    """
        Instances for evluating the NLL loss on the validation set with different temperatures and interpolation ratios
    """
    def __init__(self, net, valid_loader, src_net=None, device=None, ad=False, surro_net=None, config=None):
        super(ModelWithTemperature, self).__init__()
        self.net = net.to(device)
        if src_net is None:
            self.src_net = net
        else:
            self.src_net = src_net
        self.surro_net = surro_net
        self.config = config

        # store logits
        print('-- Get logits on validation set..')
        self.logits, self.labels = get_logits(self.net, valid_loader, src_net=self.src_net, ad=ad, config=self.config)
        
        if self.surro_net is not None:
            print('-- Get logits on validation set for surrogate model..')
            # self.surro_logits, self.surro_labels = get_logits(self.surro_net, valid_loader, src_net=None, ad=ad, config=self.config)
            self.surro_logits, self.surro_labels = get_logits(self.surro_net, valid_loader, src_net=net, ad=ad, config=self.config)

    def eval(self, temperature, coefficient=1, verbose=True):
        nll_criterion = nn.NLLLoss().cuda() 
        temperature = torch.tensor([temperature]).cuda()
        nll = nll_criterion(self.get_log_probas(temperature, coefficient), self.labels).item()
        if verbose:
            print('Temperature: %.3f Coefficient: %.3f NLL: %.3f' % (temperature.item(), coefficient, nll))
        return nll

    def get_log_probas(self, temperature, coefficient=1):
        probas = F.softmax(self.logits / temperature, dim=1)
        if coefficient >= 0 and coefficient < 1:
            probas *= coefficient
            probas_surrogate = F.softmax(self.surro_logits / temperature, dim=1)
            probas += probas_surrogate * (1 - coefficient)
        elif coefficient == 1:
            pass
        else:
            raise ValueError('Invalid coefficient: %g!' % coefficient)
        return torch.log(probas + 1e-8)
    

class AccuracyWithTemperature(nn.Module):
    """
        Instances for evluating the accuracy of the alternative labeling with different temperatures and interpolation ratios
    """
    def __init__(self, net, train_loader, device=None, ad=False, num_classes=10, config=None):
        super(AccuracyWithTemperature, self).__init__()
        self.net = net.to(device)
        self.num_classes = num_classes
        
        self.config = config
        
        # store logits
        print('-- Get logits on training set..')
        self.logits, self.labels = get_logits(self.net, train_loader, ad=ad, config=self.config)

    def eval_accuracy(self, temperature=2, coefficient=1):
        probas = self.get_kd_probas(temperature=temperature, coefficient=coefficient)
        _, preds = probas.max(1)
        return (preds.eq(self.labels).sum() / self.labels.size(0)).item()      
    
    def get_kd_probas(self, temperature=2, coefficient=1):
        probas = F.softmax(self.logits / temperature, dim=1)
        if coefficient > 0 and coefficient < 1:
            labels_one_hot = F.one_hot(self.labels, num_classes=self.num_classes).float()
            probas *= coefficient
            probas += labels_one_hot * (1 - coefficient)
        elif coefficient == 1:
            pass
        else:
            raise ValueError('Invalid coefficient: %g!' % coefficient)
        return probas


def main():
    # -- hypers
    # path = 'checkpoints/svhn_sgd_PreActResNet18_ad_pgd_10_alpha=1_cosine_lr=0.01_wd=0_0005_mom=0_9_pgd_10_nval=7325'
    # path = 'checkpoints/svhn_sgd_PreActResNet18_ad_pgd_10_alpha=1_lr=1e-02_wd=0_0005_mom=0_9_pgd_10_nval=7325-1'
    # path = 'checkpoints/cifar100_sgd_wrn-40-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9-2'
    # path = 'checkpoints/cifar100_sgd_wrn-40-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_eval=swa_start_at_150'
    # path = 'checkpoints/cifar100_sgd_wrn-40-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_crl4=swa_alpha=1_T=1_linear_eval=swa_start_at_150_non_determine'
    path = 'checkpoints/cifar100_sgd_wrn-40-2_swa_at_150_lr=5e-02_bs=64_wd=0_0005_mom=0_9_mixup=0_2'
    dataset = 'cifar100' # svhn' # 'cifar10'
    model = 'wrn' # PreActResNet18'
    depth = 40 # 18
    width = 2 # 10
    state = 'best_model_loss'
    max_iter = None # 10 # 100 # None
    gpu_id = '0'
    valset = 'testloader' # 'valloader'

    acc_min = None # 0.9 # Minimum accuracy of the alternative labeling. if none, no constraint on the accuracy
    ad = False # True # If evaluate on adversarial examples
    interpolation = False # True # If interpolating probability with the given labels
    # state_surro = 'model' # if interpolation, provide surro model

    # -- env
    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # config
    config = {'dataset': dataset,
              'soft_label': False,
              'soft_label_test': False,
              # 'valsize': 7325,
              'shuffle_train_loader': False,
              'random_augment': False,
              'adversary': 'pgd', # if evaluate on adversarial examples, we use PGD attack
              'eps': 16, # 8, 
              'pgd_alpha': 4, # 2,
              'pgd_iter': 10,
              'randomize': True,
              'device': device,
              'max_iter': max_iter,
             }
    config = Dict2Obj(config)

    # scale epsilon
    if ad:
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
    # elif any(char.isdigit() for char in state):
    #     phase, idx = state.split('-')
    #     model_state = '%s_model-%s.pt' % (phase, idx)
    else:
        model_state = '%s.pt' % state
        # raise KeyError(state)
    print(model_state)
    net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state=model_state, device=device)
    if interpolation:
        model_state_surro = '%s.pt' % state_surro
        print(model_state_surro)
        surro_net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel, model=model, depth=depth, width=width, state=model_state_surro, device=device)
    else:
        surro_net = None

    # -- Grid search - 2D
    print('>>>>>>>>>>> Initialize evaluation instance..')
    scaled_model = ModelWithTemperature(net, valloader, surro_net=surro_net, device=device, ad=ad, config=config)
    # acc_model = AccuracyWithTemperature(net, trainloader, num_classes=loaders.num_classes, device=device, ad=ad, config=config)

    print('>>>>>>>>>>> Start grid searching for optimal hyperparameters..')
    temperatures = np.exp(np.linspace(np.log(0.2), np.log(16), 100))
    if interpolation:
        coefficients = np.linspace(0.01, 1.0, 100)
    else:
        coefficients = np.array([1])
    nlls = np.zeros((len(temperatures), len(coefficients)))
    accs = np.zeros((len(temperatures), len(coefficients)))

    for i, temperature in enumerate(temperatures):
        for j, coefficient in enumerate(coefficients):
            print('T = %g, rho = %g' % (temperature, coefficient), end='\r')
            nlls[i, j] = scaled_model.eval(temperature=temperature,
                                           coefficient=coefficient,
                                           verbose=False)

            # accs[i, j] = acc_model.eval_accuracy(temperature=temperature,
            #                                      coefficient=coefficient)

    if acc_min is None:
        idx_min = np.unravel_index(np.argmin(nlls), nlls.shape)
        temperature_optimal = temperatures[idx_min[0]]
        coefficient_optimal = coefficients[idx_min[1]]
        nll_optimal = nlls[idx_min]
        acc_optimal = accs[idx_min]
    else:
        nlls_masked = np.array(nlls)
        nlls_masked[accs < acc_min] = np.max(nlls_masked) * 2
        idx_min = np.unravel_index(np.argmin(nlls_masked), nlls_masked.shape)
        temperature_optimal = temperatures[idx_min[0]]
        coefficient_optimal = coefficients[idx_min[1]]
        nll_optimal = nlls_masked[idx_min]
        acc_optimal = accs[idx_min]

    print('T* = %.2f, rho* = %.2f, nll_min = %.4f, acc_train = %.3f' % (temperature_optimal, coefficient_optimal, nll_optimal, acc_optimal))
    # logger = Logger('%s/log_optimal_parameter_%s.txt' % (path, state))
    # logger.set_names(['T*', 'rho*', 'nll_min', 'acc'])
    # logger.append([temperature_optimal, coefficient_optimal, nll_optimal, acc_optimal])
    # logger.close()

    # print('>>>>>>>>>>> save results..')
    # save_data = {'temperature': temperatures,
    #              'ratio': coefficients,
    #              'nll': nlls,
    #              'accs': accs}
    # torch.save(save_data, '%s/log_optimal_parameter_%s.pt' % (path, state))

    preds = scaled_model.logits.max(1)[1]
    corrects = preds.eq(scaled_model.labels.view_as(preds))
    print(corrects.size())
    print('acc test = %.4f' % corrects.float().mean())

    metrics = ['NLL', 'Brier', 'ECE', 'AURC']
    conf = Confidence(metrics=metrics, num_classes=loaders.num_classes, device=device)
    print('--- Original ---')
    nll = conf.nll(scaled_model.logits, scaled_model.labels)
    ece = conf.ece(F.softmax(scaled_model.logits, dim=1), corrects, bins=15)
    aurc = conf.aurc(F.softmax(scaled_model.logits, dim=1), corrects)
    print('NLL = %.4f  ECE = %.4f AURC = %.4f' %  (nll, ece, aurc))
    print('--- Calibrated ---')
    nll = conf.nll(scaled_model.logits / temperature_optimal, scaled_model.labels)
    ece = conf.ece(F.softmax(scaled_model.logits / temperature_optimal, dim=1), corrects, bins=15)
    aurc = conf.aurc(F.softmax(scaled_model.logits / temperature_optimal, dim=1), corrects)
    print('NLL = %.4f  ECE = %.4f AURC = %.4f' %  (nll, ece, aurc))


if __name__ == '__main__':

    main()
