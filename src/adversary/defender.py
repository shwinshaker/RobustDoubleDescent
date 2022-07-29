#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Logger, AverageMeter, accuracy
from ..utils import AvgTracker
from . import attack, scale_step
from .loss import trades_loss

from ..utils import mse_one_hot, ce_soft, is_parallel

import time
import copy
import os

__all__ = ['AdTrainer']

class AdTrainer:
    def __init__(self, loaders, net, optimizer, criterion=None, config=None, time_start=None, swaer=None):
        self.loaders = loaders
        self.net = net
        self.optimizer = optimizer # only used for getting current learning rate
        self.criterion = criterion
        self.config = config
        self.device = self.config.device
        self.time_start = time_start

        # target
        self.target = None
        if config.target is not None:
            self.target = loaders.class_to_idx[config.target]

        # scale epsilon (each channel is different because different range)
        ## save unscaled eps for auto attack
        config.eps_ = config.eps
        config.pgd_alpha_ = config.pgd_alpha
        config.eps = scale_step(config.eps, config.dataset, device=config.device)
        config.pgd_alpha = scale_step(config.pgd_alpha, config.dataset, device=config.device)
        print('scaled eps [train]:', config.eps, config.pgd_alpha)

        # external model
        if config.ext_model:
            self.net_ext = copy.deepcopy(self.net)
            state_dict = torch.load(config.ext_model, map_location=config.device)
            self.net_ext.load_state_dict(state_dict)
            self.net_ext.eval()

        # sanity check and setup loss function
        self.epoch = config.epoch_start
        self.__ad_setup()

    def update(self, epoch, i):
        # make some logs

        if self.extra_metrics:
            self.extraLog.step(epoch, i)

        if self.config.ext_model:
            self.extmodelLog.step(epoch, i)

        ## epoch is current epoch + 1
        self.epoch = epoch + 1

    def reset(self, epoch):
        assert(epoch == self.epoch - 1), 'reset is not called after update!'

        ## reset some logger
        if self.extra_metrics:
            self.extraLog.reset()

        if self.config.ext_model:
            self.extmodelLog.reset()

    def close(self):
        if self.extra_metrics:
            self.extraLog.close()

        if self.config.ext_model:
            self.extmodelLog.close()

    def _loss(self, inputs, labels, weights, epoch=None):
        # template
        pass

    def __ad_setup(self):

        self.extra_metrics = []

        if not self.config.adversary:
            self._loss = self._clean_loss

            if hasattr(self.config, 'kd') and self.config.kd:
                self._loss = self._clean_loss_kd

                assert(self.config.kd_coeff_st > 0)
                if isinstance(self.config.kd_teacher_st, str):
                    self.teacher_st = copy.deepcopy(self.net)
                    self.teacher_st.load_state_dict(torch.load(self.config.kd_teacher_st, map_location=self.config.device))
                    self.teacher_st.eval()
                elif isinstance(self.config.kd_teacher_st, list):
                    self.teacher_st = []
                    for path in self.config.kd_teacher_st:
                        net_teacher = copy.deepcopy(self.net)
                        net_teacher.load_state_dict(torch.load(path, map_location=self.config.device))
                        net_teacher.eval()
                        self.teacher_st.append(net_teacher)
                else:
                    raise KeyError('Missing kd teacher.')

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            return

        if self.config.adversary in ['gaussian', 'fgsm', 'pgd', 'aa']:
            self._loss = self._ad_loss
            if hasattr(self.config, 'kd') and self.config.kd:
                self._loss = self._ad_loss_kd

                if self.config.kd_teacher_st:
                    assert(self.config.kd_coeff_st > 0)
                    self.teacher_st = copy.deepcopy(self.net)
                    self.teacher_st.load_state_dict(torch.load(self.config.kd_teacher_st, map_location=self.config.device))
                    self.teacher_st.eval()
                else:
                    assert(self.config.kd_coeff_st == 0)

                if self.config.kd_teacher_rb:
                    assert(self.config.kd_coeff_rb > 0)
                    if isinstance(self.config.kd_teacher_rb, str):
                        self.teacher_rb = copy.deepcopy(self.net)
                        if is_parallel('.'):
                            self.teacher_rb.module.load_state_dict(torch.load(self.config.kd_teacher_rb, map_location=self.config.device))
                        else:
                            self.teacher_rb.load_state_dict(torch.load(self.config.kd_teacher_rb, map_location=self.config.device))
                        self.teacher_rb.eval()
                    elif isinstance(self.config.kd_teacher_rb, list):
                        self.teacher_rb = []
                        for path in self.config.kd_teacher_rb:
                            net_teacher = copy.deepcopy(self.net)
                            net_teacher.load_state_dict(torch.load(path, map_location=self.config.device))
                            net_teacher.eval()
                            self.teacher_rb.append(net_teacher)
                else:
                    assert(self.config.kd_coeff_rb == 0)

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc', 'Train-Loss-Ad', 'Train-Acc-Ad']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            # log for external model evaluation
            if self.config.ext_model:
                self.extmodelLog = AvgTracker('log_ext_model',
                                              self.optimizer,
                                              metrics=['Train-Loss-Ad', 'Train-Acc-Ad'],
                                              time_start=self.time_start,
                                              config=self.config)

            return

        # No extra logger: 'false' training accuracy not recorded for integrated loss - needs to code within specific loss function
        # other things not supported currectly..
        if self.config.target:
            raise NotImplementedError('Targeted attack not supported! TODO..')
        # if hasattr(self.config, 'alpha_sample_path') and self.config.alpha_sample_path:
        #     # for trades, this was implemented, but try incorporate into the loss function
        #     raise NotImplementedError('Sample-wise trading not supported! TODO..')
        if hasattr(self.config, 'reg_sample_path') and self.config.reg_sample_path:
            raise NotImplementedError('Sample-wise regularization Not supported!')


        if hasattr(self.config, 'kd') and self.config.kd:
            raise NotImplementedError('Knowledge distillation not supported for this ad method! TODO..')

        raise KeyError('Unexpected adversary %s' % self.config.adversary)


    def _clean_loss_kd(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)
        with torch.no_grad():
            if isinstance(self.config.kd_teacher_st, str):
                outputs_st = self.teacher_st(inputs)
                probas_st = F.softmax(outputs_st / self.config.kd_temperature, dim=1)
            elif isinstance(self.config.kd_teacher_st, list):
                probas_st = torch.zeros_like(outputs)
                for teacher in self.teacher_st:
                    outputs_st = teacher(inputs)
                    probas_st += F.softmax(outputs_st / self.config.kd_temperature, dim=1)
                probas_st /= len(self.teacher_st)
            else:
                pass

        ## -- conventional kd
        loss = self.criterion(outputs, labels)
        loss_kd_st = ce_soft(temperature=self.config.kd_temperature, num_classes=self.loaders.num_classes)(outputs, probas_st)

        loss = loss * (1. - self.config.kd_coeff_st)
        loss += loss_kd_st * self.config.kd_coeff_st

        # -------- extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        return loss.mean()


    def _clean_loss(self, inputs, labels, weights, epoch=None, update=True):
        outputs = self.net(inputs)
        if 'reg' in weights:
            loss = self.criterion(outputs,
                                  labels,
                                  weights=weights['reg'].to(self.device))
        else:
            loss = self.criterion(outputs, labels)

        if not update:
            # Only functional call 
            return loss
    
        # extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        return loss.mean()

    def _ad_loss_kd(self, inputs, labels, weights, epoch=None):

        self.net.eval()
        inputs_ad = attack(self.net,
                           self.__get_inner_max_ctr(),
                           inputs, labels, weight=None,
                           adversary=self.config.adversary,
                           eps=self.config.eps,
                           pgd_alpha=self.config.pgd_alpha,
                           pgd_iter=self.config.pgd_iter,
                           randomize=self.config.rand_init,
                           target=self.target,
                           config=self.config)
        self.net.train()
        outputs_ad = self.net(inputs_ad)
        with torch.no_grad():
            if self.config.kd_teacher_st:
                outputs_st = self.teacher_st(inputs_ad)
                probas_st = F.softmax(outputs_st / self.config.kd_temperature, dim=1)
            if isinstance(self.config.kd_teacher_rb, str):
                outputs_rb = self.teacher_rb(inputs_ad)
                probas_rb = F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
            elif isinstance(self.config.kd_teacher_rb, list):
                probas_rb = torch.zeros_like(outputs_ad)
                for teacher in self.teacher_rb:
                    outputs_rb = teacher(inputs_ad)
                    probas_rb += F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
                probas_rb /= len(self.teacher_rb)
            else:
                # no rb teacher
                pass

        ## -- conventional kd
        loss_ad = self.criterion(outputs_ad, labels)
        if self.config.kd_teacher_st:
            loss_kd_st = ce_soft(temperature=self.config.kd_temperature, num_classes=self.loaders.num_classes)(outputs_ad, probas_st)
        if self.config.kd_teacher_rb:
            loss_kd_rb = ce_soft(temperature=self.config.kd_temperature, num_classes=self.loaders.num_classes)(outputs_ad, probas_rb)

        loss = loss_ad * (1. - self.config.kd_coeff_st - self.config.kd_coeff_rb)
        if self.config.kd_teacher_st:
            loss += loss_kd_st * self.config.kd_coeff_st
        if self.config.kd_teacher_rb:
            loss += loss_kd_rb * self.config.kd_coeff_rb
        loss = loss.mean()


        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))
        return loss

    def _ad_loss(self, inputs, labels, weights, epoch=None):

        # -------- clean loss
        loss = 0.
        # if pure ad loss and sample-wise alpha not enabled, don't have to do this part
        if self.config.alpha < 1.0 or 'alpha' in weights:
            # do we need to enable model training for clean training when doing adversarial training? Test it.
            # self.net.eval() 
            loss = self._clean_loss(inputs, labels, weights, update=False)

        # ------- ad loss
        eps_weight = None
        if 'weps' in weights:
            eps_weight = weights['weps']
            # TODO: Individualize epsilon here, removing the parameter `weight` in `attack` function

        pgd_alpha = self.config.pgd_alpha
        pgd_iter = self.config.pgd_iter
        adversary = self.config.adversary
        if 'num_iter' in weights:
            assert(self.config.adversary == 'pgd'), 'adversary %s not supported in instance-wise iteration mode!'
            adversary = 'pgd_custom'
            pgd_iter = weights['num_iter']
            pgd_alpha = self.__get_pgd_alpha(pgd_iter)

        self.net.eval()
        inputs_ad = attack(self.net,
                           self.__get_inner_max_ctr(),
                           inputs, labels, weight=eps_weight,
                           adversary=adversary,
                           eps=self.config.eps,
                           pgd_alpha=pgd_alpha,
                           pgd_iter=pgd_iter,
                           randomize=self.config.rand_init,
                           target=self.target,
                           config=self.config)
        self.net.train()

        outputs_ad = self.net(inputs_ad)

        if 'reg' in weights:
            loss_ad = self.criterion(outputs_ad,
                                     labels,
                                     weights=weights['reg'].to(self.device))
        else:
            loss_ad = self.criterion(outputs_ad, labels)
            # print('bce!')
            # loss_ad = bce_loss(outputs_ad, labels, reduction='none')

        # -------- combine two loss
        if 'alpha' in weights:
            # sample-wise weighting
            assert(loss.size(0) == inputs.size(0)), (loss.size(0), inputs.size(0))
            alpha = weights['alpha'].to(self.device)
            assert(loss.size() == loss_ad.size() == alpha.size()), (loss.size(), loss_ad.size(), alpha.size())
        else:
            alpha = self.config.alpha

        if 'lambda' in weights:
            lmbd = weights['lambda'].to(self.device)
        else:
            lmbd = torch.ones(inputs.size(0)).to(self.device)

        assert(loss_ad.size() == lmbd.size()), (loss_ad.size(), lmbd.size())
        loss *= (1 - alpha)
        loss += alpha * loss_ad 
        loss *= lmbd / lmbd.sum() # per-sample weight
        loss = loss.sum()
        # print(loss)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        return loss

    def __get_pgd_alpha(self, pgd_iters, acc_radius=20.):
        pgd_alphas = acc_radius / pgd_iters.float()
        pgd_alphas = pgd_alphas.view(pgd_alphas.size(0), 1, 1, 1).to(self.device)
        pgd_alphas = scale_step(pgd_alphas, dataset=self.config.dataset, device=self.device)
        return pgd_alphas

    def __get_inner_max_ctr(self):
        # Don't change the criterion in adversary generation part -- maybe change it later
        # ctr = nn.CrossEntropyLoss() 
        if not hasattr(self.config, 'ad_soft_label'):
            self.config.ad_soft_label = False
        ctr = ce_soft(num_classes=self.loaders.num_classes,
                      soft_label=self.config.ad_soft_label)
        if hasattr(self.config, 'loss'):
            if self.config.loss == 'ce':
                pass
            elif self.config.loss == 'mse':
                ctr = mse_one_hot(num_classes=self.loaders.num_classes,
                                  soft_label=self.config.ad_soft_label)
            else:
                raise NotImplementedError()
        return ctr

    def _trades_loss(self, inputs, labels, weights, epoch=None):
        # note: The official implementation use CE + KL * beta - amounts to alpha~= 0.85
        #       Previously we use (1-alpha) * CE + alpha * KL
        # integrate clean loss in trades loss
        # sample-weighting in trades loss - later
        loss, outputs_ad, inputs_ad = trades_loss(self.net, inputs, labels, weights,
                                                  eps=self.config.eps,
                                                  alpha=self.config.pgd_alpha,
                                                  num_iter=self.config.pgd_iter,
                                                  norm='linf',
                                                  rand_init=self.config.rand_init,
                                                  config=self.config)

        # -------- KD
        if hasattr(self.config, 'kd') and self.config.kd:
            with torch.no_grad():
                if self.config.kd_teacher_st:
                    outputs_st = self.teacher_st(inputs_ad)
                    probas_st = F.softmax(outputs_st / self.config.kd_temperature, dim=1)
                if isinstance(self.config.kd_teacher_rb, str):
                    outputs_rb = self.teacher_rb(inputs_ad)
                    probas_rb = F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
                elif isinstance(self.config.kd_teacher_rb, list):
                    probas_rb = torch.zeros_like(outputs_ad)
                    for teacher in self.teacher_rb:
                        outputs_rb = teacher(inputs_ad)
                        probas_rb += F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
                    probas_rb /= len(self.teacher_rb)
                else:
                    # no rb teacher
                    pass

            loss = loss * (1. - self.config.kd_coeff_st - self.config.kd_coeff_rb)
            if self.config.kd_teacher_st:
                loss_kd_st = ce_soft(temperature=self.config.kd_temperature, num_classes=self.loaders.num_classes)(outputs_ad, probas_st)
                loss += loss_kd_st * self.config.kd_coeff_st
            if self.config.kd_teacher_rb:
                loss_kd_rb = ce_soft(temperature=self.config.kd_temperature, num_classes=self.loaders.num_classes)(outputs_ad, probas_rb)
                loss += loss_kd_rb * self.config.kd_coeff_rb

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        return loss

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]

