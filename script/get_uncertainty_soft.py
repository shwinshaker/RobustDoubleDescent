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


def get_logits(net, loader, src_net=None, ad=False, config=None, max_iter=100, soft_label=False):
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
        with torch.no_grad():
            logits = net(inputs)
        logits_list.append(logits)
        labels_list.append(labels)
        
        ## Check acc
        _, preds = logits.max(1)
        if soft_label:
            _, true_labels = labels.max(1)
            corrects += torch.sum(preds.eq(true_labels))
        else:
            corrects += torch.sum(preds.eq(labels))
        counts += inputs.size(0)
        print('[%i/%i] acc: %.4f' % (e + 1, len(loader), corrects.float()/counts), end='\r')
    
        # to speed up, no need to check all examples
        ## but better to check all for validation loader
        if e > max_iter: 
            break

    logits_ = torch.cat(logits_list).cuda()
    labels_ = torch.cat(labels_list).cuda()
    return logits_, labels_

class ModelWithTemperature(nn.Module):
    """
        Instances for evluating the NLL loss on the validation set with different temperatures and interpolation ratios
    """
    def __init__(self, net, loader, src_net=None, device=None, soft_label=False):
        super(ModelWithTemperature, self).__init__()
        self.soft_label = soft_label
        self.net = net.to(device)
        if src_net is None:
            self.src_net = net
        else:
            self.src_net = src_net
        
        # config
        config = {'dataset': 'cifar10h',
                  'device': device,
                  'randomize': True,
                 }
        self.config = Dict2Obj(config)
        
        # store logits
        self.logits, self.labels = get_logits(self.net, loader, src_net=self.src_net, config=self.config, soft_label=soft_label)
        
    def eval(self, temperature, coefficient=1, verbose=True):
        probas = self.get_log_probas(temperature, coefficient)
        if not self.soft_label:
            nll_criterion = nn.NLLLoss().cuda() 
            nll = nll_criterion(probas, self.labels).item()
            if verbose:
                print('Temperature: %.3f Coefficient: %.3f NLL: %.3f' % (temperature.item(), coefficient, nll))
            return nll
        
        kl = F.kl_div(probas, self.labels, reduction='batchmean').item()
        if verbose:
            print('Temperature: %.3f Coefficient: %.3f NLL: %.3f' % (temperature.item(), coefficient, kl))
        return kl

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
    

# class AccuracyWithTemperature(nn.Module):
#     """
#         Instances for evluating the accuracy of the alternative labeling with different temperatures and interpolation ratios
#     """
#     def __init__(self, net, train_loader, device=None, ad=False, num_classes=10):
#         super(AccuracyWithTemperature, self).__init__()
#         self.net = net.to(device)
#         self.num_classes = num_classes
        
#         # config
#         config = {'dataset': 'cifar10',
#                   'device': device,
#                   'eps': 8, 
#                   'pgd_alpha': 2,
#                   'pgd_iter': 10,
#                   'adversary': 'pgd',
#                   'randomize': True,
#                  }
#         self.config = Dict2Obj(config)
        
#         # scale epsilon
#         self.config.eps = scale_step(self.config.eps,
#                                      dataset=self.config.dataset,
#                                      device=self.config.device)
#         self.config.pgd_alpha = scale_step(self.config.pgd_alpha,
#                                            dataset=self.config.dataset,
#                                            device=self.config.device)
        
#         # store logits
#         self.logits, self.labels = get_logits(self.net, train_loader, ad=ad, config=self.config)

#     def eval_accuracy(self, temperature=2, coefficient=1):
#         probas = self.get_kd_probas(temperature=temperature, coefficient=coefficient)
#         _, preds = probas.max(1)
#         return (preds.eq(self.labels).sum() / self.labels.size(0)).item()      
    
#     def get_kd_probas(self, temperature=2, coefficient=1):
#         probas = F.softmax(self.logits / temperature, dim=1)
#         if coefficient > 0 and coefficient < 1:
#             labels_one_hot = F.one_hot(self.labels, num_classes=self.num_classes).float()
#             probas *= coefficient
#             probas += labels_one_hot * (1 - coefficient)
#         elif coefficient == 1:
#             pass
#         else:
#             raise ValueError('Invalid coefficient: %g!' % coefficient)
#         return probas


def main():
    # -- hypers
    # path = 'checkpoints/cifar10h_sgd_wrn-16-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=id_cifar10h_tv_nsplit=5_high_quality=3600'
    path = 'checkpoints/cifar10h_sgd_wrn-16-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=id_cifar10h_tv_nsplit=5_low_quality=5400'
    dataset = 'cifar10h'
    model = 'wrn'
    depth = 16
    width = 2
    state = 'best'
    gpu_id = '1'

    # acc_min = None # 0.9 # Minimum accuracy of the alternative labeling. if none, no constraint on the accuracy
    # ad = False # If evaluate on adversarial examples
    # interpolation = False # If interpolating probability with the given labels

    # -- env
    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for soft_label in [True, False]:
        if soft_label:
            print('>>>>>>>>>>> soft label..')
        else:
            print('>>>>>>>>>>> one-hot label')
        # -- get loader
        print('>>>>>>>>>>> get loader..')
        loaders = get_loaders(dataset=dataset,
                              shuffle_train_loader=False, 
                              data_dir='/home/chengyu/Initialization/data',
                              config=Dict2Obj({'soft_label': soft_label, random_augment=False}))

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

        # -- Grid search - 2D
        print('>>>>>>>>>>> Initialize evaluation instance..')
        scaled_model_test = ModelWithTemperature(net, loaders.testloader, device=device, soft_label=soft_label)
        scaled_model_train = ModelWithTemperature(net, loaders.trainloader, device=device, soft_label=soft_label)
        # acc_model = AccuracyWithTemperature(net, loaders.trainloader, num_classes=loaders.num_classes, device=device)

        print('>>>>>>>>>>> Start grid searching for optimal hyperparameters..')
        temperatures = np.exp(np.linspace(np.log(0.2), np.log(16), 100))
        nlls_test = np.zeros(len(temperatures))
        nlls_train = np.zeros(len(temperatures))
        # accs = np.zeros(len(temperatures))

        for i, temperature in enumerate(temperatures):
            print('T = %g' % (temperature), end='\r')
            nlls_test[i] = scaled_model_test.eval(temperature=temperature,
                                                  coefficient=1,
                                                  verbose=False)

            nlls_train[i] = scaled_model_train.eval(temperature=temperature,
                                                    coefficient=1,
                                                    verbose=False)

            # accs[i] = acc_model.eval_accuracy(temperature=temperature,
            #                                      coefficient=1)

        idx_min = np.argmin(nlls_test)
        temperature_optimal = temperatures[idx_min]
        nll_optimal = nlls_test[idx_min]
        # acc_optimal = accs[idx_min]
        # print('T* = %.2f, nll_min = %.2f, acc = %.2f' % (temperature_optimal, nll_optimal, acc_optimal))
        print('T* = %.2f, nll_min = %.2f' % (temperature_optimal, nll_optimal))
        # logger = Logger('%s/log_optimal_parameter_%s.txt' % (path, state))
        # logger.set_names(['T*', 'rho*', 'nll_min', 'acc'])
        # logger.append([temperature_optimal, nll_optimal, acc_optimal])
        # logger.close()

        print('>>>>>>>>>>> save results..')
        save_data = {'temperature': temperatures,
                     'nll_test': nlls_test,
                     'nll_train': nlls_train}
        if soft_label:
            torch.save(save_data, '%s/log_uncertainty_softlabel_%s.pt' % (path, state))
        else:
            torch.save(save_data, '%s/log_uncertainty_%s.pt' % (path, state))


if __name__ == '__main__':

    main()
