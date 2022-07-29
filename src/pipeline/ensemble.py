#!./env python

import torch
import torch.nn as nn
from ..utils import test, Logger, nan_filter
from ..models import get_net
from ..adversary import ad_test
from ..analyses import load_log
from ..utils import mse_one_hot

import time
import copy
import numpy as np
import os

__all__ = ['SWA']

def get_best_acc(path='.', option='robust'):
    stats = load_log(os.path.join(path, 'log_swa.txt'), window=1)
    if option == 'robust':
        return np.max(nan_filter(stats['Test-Acc-Ad']))
    if option == 'clean':
        return np.max(nan_filter(stats['Test-Acc']))
    raise KeyError(option)

def get_swa_count(path='.'):
    stats = load_log(os.path.join(path, 'log_swa.txt'), window=1)
    return int(stats['SWA-Count'][-1])

def get_last_time(path='.'):
    return load_log(os.path.join(path, 'log_swa.txt'))['Time-elapse(Min)'][-1]


class SWA:
    """
        Stochastic Weight Averaing
    """
    def __init__(self, loaders, net, optimizer, config, time_start):

        self.loaders = loaders
        self.net = net
        # self.criterion = criterion
        # Criterion in testing is not allowed to change
        self.criterion = nn.CrossEntropyLoss()
        if hasattr(config, 'loss'):
            if config.loss == 'ce':
                pass
            elif config.loss == 'mse':
                # self.criterion = nn.MSELoss()
                self.criterion = mse_one_hot(num_classes=loaders.num_classes)
            else:
                raise NotImplementedError()
        self.optimizer = optimizer
        self.config = config

        # swa setup
        self.swa_start = config.swa_start
        self.swa_interval = config.swa_interval
        self.swa_count = 0 # number of averaged models === number of epochs since swa started
        self.swa_net = get_net(config, loaders)

        # Caution! Eps needs to be scaled. Presumably scaled in Tester

        if config.ad_test == 'aa':
            # use alternative tester
            if hasattr(config, 'class_eval') and config.class_eval:
                raise NotImplementedError('Per class evaluation not supported in aa..')
            self.__aa_setup()

        # basic logs 
        base_names = ['Epoch', 'lr', 'Time-elapse(Min)', 'SWA-Count']
        self.logger = Logger('log_swa.txt', title='log', resume=config.resume)
        metrics = ['Train-Loss', 'Test-Loss',
                   'Train-Loss-Ad', 'Test-Loss-Ad',
                   'Train-Acc', 'Test-Acc',
                   'Train-Acc-Ad', 'Test-Acc-Ad']
        self.logger.set_names(base_names + metrics)

        ## set up time
        self.time_start = time_start
        self.last_end = 0.
        if config.resume:
            self.last_end = get_last_time() # min

        # save best model
        self.best = config.best
        self.best_acc = 0.
        if config.resume:
            self.best_acc = get_best_acc(option=self.best)
            print('[SWA] > Best: %.2f' % (self.best_acc))

            self.swa_count = get_swa_count()
            state_dict = torch.load('./model_swa.pt')
            self.swa_net.load_state_dict(state_dict)

    def __swa_update(self):
        # Init swa model will be replaced by the model at the first update step, since the ratio is 1
        self._moving_average(self.swa_net, self.net, 1.0 / (self.swa_count + 1))
        self.swa_count += 1
        self._bn_update(self.loaders.trainloader, self.swa_net)
        print('[SWA] # SWA update steps: %i' % self.swa_count)

    def _moving_average(self, net1, net2, alpha=1):
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha

    def _bn_update(self, loader, model):
        """
            BatchNorm buffers update (if any).
            Performs 1 epochs to estimate buffers average using train dataset.

            :param loader: train dataset loader for buffers average estimation.
            :param model: model being update
            :return: None
        """
        if not self._check_bn(model):
            return
        model.train()
        momenta = {}
        model.apply(self._reset_bn)
        model.apply(lambda module: self._get_momenta(module, momenta))
        n = 0
        for input, _, _ in loader:
            input = input.cuda()
            b = input.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input)
            n += b

        model.apply(lambda module: self._set_momenta(module, momenta))

    def _check_bn(self, model):

        def __check_bn(module, flag):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                flag[0] = True

        flag = [False]
        model.apply(lambda module: __check_bn(module, flag))
        return flag[0]
    
    def _reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def _get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
    
    def _set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def __update_best(self, epoch, test_acc, test_acc_ad):
        if self.best.lower() == 'robust':
            acc = test_acc_ad
        else:
            acc = test_acc

        if acc > self.best_acc:
            print('[SWA] > SWA Best got at epoch %i. Best: %.2f Current: %.2f' % (epoch, acc, self.best_acc))
            self.best_acc = acc
            torch.save(self.swa_net.state_dict(), 'best_model_swa.pt')

    def update(self, epoch):
        if not (epoch > self.swa_start and (epoch - self.swa_start) % self.swa_interval == 0):
            return

        assert(not self.net.training), 'Model is not in evaluation mode, calling from wrong place!'

        # swa updated
        self.__swa_update()
        torch.save(self.swa_net.state_dict(), 'model_swa.pt')
        self.swa_net.eval()

        # train - test
        train_loss, train_prec1, train_ex_metrics = 0, 0, dict()
        train_loss_ad, train_prec1_ad, train_ex_metrics_ad = 0, 0, dict()
        if self.config.traintest:
            train_loss, train_prec1, train_ex_metrics = self.__test(self.loaders.traintestloader)
            if not self.config.ad_test:
                train_loss_ad, train_prec1_ad, train_ex_metrics_ad = 0, 0, dict()
            elif self.config.ad_test == 'aa':
                train_loss_ad, train_prec1_ad, train_ex_metrics_ad = self.__ad_test_aa(mode='train')
            elif self.config.ad_test in ['fgsm', 'pgd']:
                train_loss_ad, train_prec1_ad, train_ex_metrics_ad = self.__ad_test(self.loaders.traintestloader)
            else:
                raise KeyError('Adversary %s not supported!' % self.config.ad_test)

        # test
        test_loss, test_prec1, test_ex_metrics = self.__test(self.loaders.testloader)
        if not self.config.ad_test:
            test_loss_ad, test_prec1_ad, test_ex_metrics_ad = 0, 0, dict()
        elif self.config.ad_test == 'aa':
            test_loss_ad, test_prec1_ad, test_ex_metrics_ad = self.__ad_test_aa(mode='test')
        elif self.config.ad_test in ['fgsm', 'pgd']:
            test_loss_ad, test_prec1_ad, test_ex_metrics_ad = self.__ad_test(self.loaders.testloader)
        else:
            raise KeyError('Adversary %s not supported!' % self.config.ad_test)

        # best
        self.__update_best(epoch, test_prec1, test_prec1_ad)

        # logs
        time_elapse = (time.time() - self.time_start)/60 + self.last_end
        logs_base = [epoch, self.__get_lr(), time_elapse, self.swa_count]
        logs = [_ for _ in logs_base]
        logs += [train_loss, test_loss,
                 train_loss_ad, test_loss_ad,
                 train_prec1, test_prec1,
                 train_prec1_ad, test_prec1_ad]
        self.logger.append(logs)

    def close(self):
        self.logger.close()

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]
    
    def __test(self, loader):
        return test(loader, self.swa_net, self.criterion, self.config, classes=self.loaders.classes)

    def __ad_test(self, loader):
        return ad_test(loader, self.swa_net, self.criterion, self.config, classes=self.loaders.classes)

    def __aa_setup(self):
        # evaluate on a random subset fixed through training
        self.aa_attacker = AAAttacker(net=self.swa_net,
                                      normalize=True,
                                      mode='fast',
                                      sample=1000,
                                      rand_sample=False,
                                      seed=7,
                                      log_path=None,
                                      device=self.config.device,
                                      data_dir=self.config.data_dir)

    def __ad_test_aa(self, mode='test'):
        return 0., self.aa_attacker.evaluate(mode=mode)[1], dict()

