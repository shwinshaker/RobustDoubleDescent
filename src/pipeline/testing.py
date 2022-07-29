#!./env python

import torch
import torch.nn as nn
from ..utils import test, Logger, nan_filter
from ..utils import save_checkpoint, save_model
from ..adversary import ad_test, scale_step
from ..adversary import AAAttacker
from ..utils import mse_one_hot, ce_soft
import time
import copy

from ..analyses import load_log
import numpy as np
import os

__all__ = ['Tester']

def get_best(path='.', option='robust', phase='Acc'):
    stats = load_log(os.path.join(path, 'log.txt'), window=1)
    if phase == 'Acc':
        extrema = np.max
    elif phase == 'Loss':
        extrema = np.min
    else:
        raise KeyError(phase)
    if option == 'robust':
        return extrema(nan_filter(stats['Test-%s-Ad' % phase]))
    if option == 'clean':
        return extrema(nan_filter(stats['Test-%s' % phase]))
    raise KeyError(option)

def get_last_time(path='.'):
    return load_log(os.path.join(path, 'log.txt'))['Time-elapse(Min)'][-1]

class Tester:
    def __init__(self, loaders, net, optimizer, config, time_start, scheduler=None):

        self.loaders = loaders
        self.net = net
        # self.criterion = criterion
        # Criterion in testing is not allowed to change
        # self.criterion = nn.CrossEntropyLoss()
        if not hasattr(config, 'ad_soft_label_test'):
            config.ad_soft_label_test = False
        self.criterion = ce_soft(num_classes=loaders.num_classes,
                                 soft_label=config.ad_soft_label_test)
        if hasattr(config, 'loss'):
            if config.loss == 'ce':
                pass
            elif config.loss == 'mse':
                print('MSE Loss used!')
                # self.criterion = nn.MSELoss()
                self.criterion = mse_one_hot(num_classes=loaders.num_classes,
                                             soft_label=config.ad_soft_label_test)
            else:
                raise NotImplementedError()

        self.optimizer = optimizer
        self.scheduler = scheduler # only for saving checkpoint!
        self.config = config

        # scale epsilon
        config.eps_test = scale_step(config.eps_test, config.dataset, device=config.device)
        config.pgd_alpha_test = scale_step(config.pgd_alpha_test, config.dataset, device=config.device)
        print('scaled eps:', config.eps_test, config.pgd_alpha_test)

        if config.ad_test == 'aa':
            # use alternative tester
            if hasattr(config, 'class_eval') and config.class_eval:
                raise NotImplementedError('Per class evaluation not supported in aa..')
            self.__aa_setup()
            # raise NotImplementedError('auto attack takes additional 8 hours..')

        # basic logs 
        base_names = ['Epoch', 'Mini-batch', 'lr', 'Time-elapse(Min)']
        self.logger = Logger('log.txt', title='log', resume=config.resume)
        metrics = ['Train-Loss', 'Test-Loss',
                   'Train-Loss-Ad', 'Test-Loss-Ad',
                   'Train-Acc', 'Test-Acc',
                   'Train-Acc-Ad', 'Test-Acc-Ad']
        self.logger.set_names(base_names + metrics)

        # -- validation logs
        if hasattr(self.loaders, 'valloader'):
            self.logger_val = Logger(os.path.join(config.save_dir, 'log_val.txt'), title='validation log', resume=config.resume)
            metrics = ['Val-Loss', 'Val-Loss-Ad', 'Val-Acc', 'Val-Acc-Ad', 'Val-Acc5', 'Val-Acc5-Ad']
            for m in config.robust_metrics:
                metrics.append('Val-%s' % m)
            self.logger_val.set_names(base_names + metrics)

            self.best_acc_val = 0.
            self.best_loss_val = np.inf
            if config.resume:
                self.best_acc_val = get_best(path=config.save_dir, option=self.best, phase='Acc', mode='val')
                self.best_loss_val = get_best(path=config.save_dir, option=self.best, phase='Loss', mode='val')
                print('> Best val Acc: %.2f Best val Loss: %.2f' % (self.best_acc_val, self.best_loss_val))

        # extra logs for class-wise evaluation
        if hasattr(config, 'class_eval') and config.class_eval:
            self.logger_c = Logger('log_class.txt', title='log for class-wise acc', resume=config.resume)
            metrics = ['Test-Acc-%s' % c for c in loaders.classes]
            if config.adversary:
                metrics.extend(['Test-Acc-%s-Ad' % c for c in loaders.classes])
            self.logger_c.set_names(base_names + metrics)

        self.time_start = time_start
        self.last_end = 0.
        if config.resume:
            self.last_end = get_last_time() # min

        # read best acc
        self.best = config.best
        self.best_acc = 0.
        self.best_loss = np.inf
        if config.resume:
            self.best_acc = get_best(option=self.best, phase='Acc')
            self.best_loss = get_best(option=self.best, phase='Loss')
            print('> Best Acc: %.2f Best Loss: %.2f' % (self.best_acc, self.best_loss))

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]
    
    def __test(self, loader):
        return test(loader, self.net, self.criterion, self.config, classes=self.loaders.classes)

    def __ad_test(self, loader):
        return ad_test(loader, self.net, self.criterion, self.config, classes=self.loaders.classes)

    def __aa_setup(self):
        # evaluate on a random subset fixed through training
        self.aa_attacker = AAAttacker(net=self.net,
                                      normalize=True,
                                      mode='fast',
                                      sample=1000,
                                      rand_sample=False,
                                      seed=7,
                                      log_path=None,
                                      dataset=self.config.dataset,
                                      device=self.config.device,
                                      data_dir=self.config.data_dir)

    def __ad_test_aa(self, mode='test'):
        ## dummy output
        return 0., self.aa_attacker.evaluate(mode=mode)[1], dict()

    def __update_best(self, epoch, test_acc, test_acc_ad, test_loss, test_loss_ad):
        if self.best.lower() == 'robust':
            acc = test_acc_ad
            loss = test_loss_ad
        else:
            acc = test_acc
            loss = test_loss

        if acc > self.best_acc:
            print('> Best acc got at epoch %i. Best: %.2f Current: %.2f' % (epoch, acc, self.best_acc))
            self.best_acc = acc
            save_model(self.net, 'best_model', self.config)

        if loss < self.best_loss:
            print('> Best loss got at epoch %i. Best: %.2f Current: %.2f' % (epoch, loss, self.best_loss))
            self.best_loss = loss
            save_model(self.net, 'best_model_loss', self.config)

    def __update_best_val(self, epoch, val_acc, val_acc_ad, val_loss, val_loss_ad):
        if self.best.lower() == 'robust':
            acc = val_acc_ad
            loss = val_loss_ad
        else:
            acc = val_acc
            loss = val_loss

        if acc > self.best_acc_val:
            print('> Best val acc got at epoch %i. Best: %.2f Current: %.2f' % (epoch, acc, self.best_acc_val))
            self.best_acc_val = acc
            save_model(self.net, 'best_model_val', self.config)

        if loss < self.best_loss_val:
            print('> Best val loss got at epoch %i. Best: %.2f Current: %.2f' % (epoch, loss, self.best_loss_val))
            self.best_loss_val = loss
            save_model(self.net, 'best_model_loss_val', self.config)

    def update(self, epoch, i):

        assert(not self.net.training), 'Model is not in evaluation mode, calling from wrong place!'
        """
            The net should stay on test mode when attack
                because it makes no sense to calculate the batch statistics of adversary examples
        """

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
        self.__update_best(epoch, test_prec1, test_prec1_ad, test_loss, test_loss_ad)

        # logs
        time_elapse = (time.time() - self.time_start)/60 + self.last_end
        logs_base = [epoch, i, self.__get_lr(), time_elapse]
        logs = [_ for _ in logs_base]
        logs += [train_loss, test_loss,
                 train_loss_ad, test_loss_ad,
                 train_prec1, test_prec1,
                 train_prec1_ad, test_prec1_ad]
        self.logger.append(logs)

        # evaluation on each class
        if hasattr(self.config, 'class_eval') and self.config.class_eval:
            logs = [_ for _ in logs_base]
            logs += test_ex_metrics['class_acc'] + test_ex_metrics_ad['class_acc']
            self.logger_c.append(logs)

        # evaluation on validation set
        if hasattr(self.loaders, 'valloader'):
            val_loss, val_prec1, val_prec5, val_ex_metrics = self.__test(net, self.loaders.valloader)
            if not self.config.ad_test:
                val_loss_ad, val_prec1_ad, val_prec5_ad, val_ex_metrics_ad = 0, 0, 0, dict()
            elif self.config.ad_test == 'aa':
                raise NotImplementedError('Validation loader not yet supported in AutoAttack..')
                # test_loss_ad, test_prec1_ad, test_ex_metrics_ad = self.__ad_test_aa(mode='val')
            elif self.config.ad_test in ['fgsm', 'pgd']:
                val_loss_ad, val_prec1_ad, val_prec5_ad, val_ex_metrics_ad = self.__ad_test(net, self.loaders.valloader)
            else:
                raise KeyError('Adversary %s not supported!' % self.config.ad_test)

            # update best
            self.__update_best_val(net, epoch, val_prec1, val_prec1_ad, val_loss, val_loss_ad)

            # logs
            logs = [_ for _ in logs_base]
            logs += [val_loss, val_loss_ad, val_prec1, val_prec1_ad, val_prec5, val_prec5_ad]
            for m in self.config.robust_metrics:
                logs.append(val_ex_metrics_ad['rb_metric'][m])
            self.logger_val.append(logs)

        # save model
        self.__save_model(self.net, epoch=epoch)


    def __save_model(self, net, suffix='', epoch=None):
        # save model for post-training process separately
        if hasattr(self.config, 'save_model_at') and self.config.save_model_at:
            if isinstance(self.config.save_model_at, list):
                if epoch in self.config.save_model_at:
                    print('Model saved at epoch %i' % epoch) 
                    save_model(net, 'model-%i%s' % (epoch, suffix), self.config)
            if isinstance(self.config.save_model_at, str):
                if epoch in range(*eval(self.config.save_model_at)):
                    print('Model saved at epoch %i' % epoch) 
                    save_model(net, 'model-%i%s' % (epoch, suffix), self.config)

        # save last model
        if epoch == self.config.epochs - 1:
            save_model(net, 'model%s' % suffix, self.config)

    def close(self):
        self.logger.close()
        if hasattr(self.config, 'class_eval') and self.config.class_eval:
            self.logger_c.close()
        if hasattr(self.loaders, 'valloader'):
            self.logger_val.close()
