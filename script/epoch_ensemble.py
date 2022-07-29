#!./env python
import torch
import numpy as np
import torch.nn as nn

import os
import time

from src.utils import Logger
from src.preprocess import get_loaders
from src.analyses import get_ad_examples, get_net
from src.utils import Dict2Obj
from src.utils import RobustTracker

## average model
class ModelAvg(nn.Module):
    def __init__(self, model1, model2, trainloader):
        super(ModelAvg, self).__init__()

        self.model = self.moving_average(model1, model2, alpha=0.) 
        # self._bn_update(trainloader, self.model)

        self._loss = nn.CrossEntropyLoss()

    def moving_average(self, net1, net2, alpha=0.5):
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
        return net1

    def forward(self, x):
        return self.model(x)
        
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
        model.eval()


class ActivationAvg(nn.Module):
    # average after log softmax - needs to alter the cross entropy loss to nll loss
    def __init__(self, model1, model2, normalize=False):
        super(ActivationAvg, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self._loss = nn.NLLLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.normalize = normalize

    def forward(self, x):
        output1 = self.logsoftmax(self.model1(x))
        output2 = self.logsoftmax(self.model2(x))
        output = (output1 + output2) / 2.
        if self.normalize:
            output = output.exp()
            output = output / output.sum(dim=1, keepdims=True)
            output = output.log()
        return output


class OutputAvg(nn.Module):
    # average before log softmax
    def __init__(self, model1, model2):
        super(OutputAvg, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
        self._loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return (self.model1(x) + self.model2(x)) / 2.


def get_ensemble(epoch, loaders, num_splits=2, device=None, width=5):
    nets = []
    for split in range(num_splits):
        # print('Model - %i' % split)
        if split == 0:
<<<<<<< Updated upstream
            # path = 'checkpoints/sgd_wrn-28-5_gain=1_0_ad_pgd_10_alpha=1_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000'
            path = '../../checkpoints/adam_wrn-28-%i_gain=1_0_ad_pgd_10_alpha=1_lr=1e-04_mom=0_9_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000' % width
        else:
            # path = 'checkpoints/sgd_wrn-28-5_gain=1_0_ad_pgd_10_alpha=1_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000_%i' % split
            path = '../../checkpoints/adam_wrn-28-%i_gain=1_0_ad_pgd_10_alpha=1_lr=1e-04_mom=0_9_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000_%i' % (width, split)
=======
            path = 'checkpoints/ad_aug_pgd_eps=8_epst=8_adam_wrn-28-1_gain=1_0_lr=1e-04_mom=0_9_sub=mean_rb_id_pgd10_epoch_rand_5000_1'
        else:
            path = 'checkpoints/ad_aug_pgd_eps=8_epst=8_adam_wrn-28-1_gain=1_0_lr=1e-04_mom=0_9_sub=mean_rb_id_pgd10_epoch_rand_5000-1'
>>>>>>> Stashed changes

        net = get_net(path, num_classes=loaders.num_classes, n_channel=loaders.n_channel,
                      model='wrn', depth=28, width=width, state='model-%i.pt' % epoch,
                      device=device)
        net.eval()
        nets.append(net)

    # net = ActivationAvg(nets[0], nets[1], normalize=True)
    net = OutputAvg(nets[0], nets[1])
    # net = ModelAvg(nets[0], nets[1], trainloader=loaders.trainloader)
    net.eval()
    check_bn(net)

    return net


def check_bn(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert(not m.training)

<<<<<<< Updated upstream
def eval_net(net, loaders, device=None):
=======
def eval_ensemble(loaders, epoch, num_splits=2, device=None, width=5, ad=True,):

    nets = get_nets(epoch, loaders, num_splits=num_splits, width=width, device=device)
    # net = ActivationAvg(nets[0], nets[1], normalize=True)
    net = OutputAvg(nets[0], nets[1])
    # net = ModelAvg(nets[0], nets[1], trainloader=loaders.trainloader)
    net.eval()
    check_bn(net)
>>>>>>> Stashed changes

    ## Evaluation
    acc = 0
    loss = 0
    count = 0
    for batch_idx, (inputs, targets, _) in enumerate(loaders.testloader):
        inputs_, targets = inputs.to(device), targets.to(device)
        if ad:
            inputs, _ = get_ad_examples(net, inputs_, labels=targets, criterion=net._loss,
                                        adversary='pgd', eps=8, pgd_alpha=2, pgd_iter=10,
                                        dataset='cifar10', device=device)
        else:
            inputs = inputs_
        with torch.no_grad():
            outputs = net(inputs)
            loss += net._loss(outputs, targets).item() * inputs.size(0)
            _, preds = outputs.max(1)
            acc += preds.eq(targets).sum().item()
            count += inputs.size(0)
        print('[%i / %i] %.4f %.4f' % (batch_idx + 1,
                                       len(loaders.testloader),
                                       (acc / count)*100,
                                       loss / count),
              end='\r')
    return acc / count * 100, loss / count


if __name__ == '__main__':

<<<<<<< Updated upstream
    gpu_id = 1
    resume = False
    width = 5
    num_splits = 2
    save_path = 'tmp/ad_ensemble'
    random_augment = True # random augment allowed, only matters for mindist as that one is evaluated on training set

    os.chdir(save_path)
=======
    gpu_id = 7
    resume = False
    width = 1
    num_splits = 2
    start_epoch = 0
    end_epoch = 1000
>>>>>>> Stashed changes

    ## Set device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ## Set config
    config = {'traintest': True,
              'dataset': 'cifar10',
              'batch_size': 128,
              'rbTrack': ['FGSM'],
              'rbTrackPhase': 'train',
              'rbTrackSubsize': 1000,
              'rbTrackSavebest': False,
              'device': device,
             }
    config = Dict2Obj(config)
    
    start = time.time()

    ## get loader
    dataset = 'cifar10'
    loaders = get_loaders(dataset=dataset, random_augment=random_augment,
                          shuffle_train_loader=False,
                          data_dir='/home/chengyu/Initialization/data',
                          config=config)

    ## Robust metric log
    net = get_ensemble(0, loaders, num_splits=num_splits, width=width, device=device)
    rbLog = RobustTracker(net, loaders, config, start)

    ## log
    file_name = 'log_acc_output_avg_wrn-%i' % width
    if num_splits != 2:
        file_name += '_split=%i' % num_splits
<<<<<<< Updated upstream
    # file_name += '_diffDataset_sgd'
    # file_name += '_alpha=0_noBNUpdate'
=======
    file_name += '_eps=8_epts=8'
>>>>>>> Stashed changes

    logger = Logger('%s.txt' % file_name, title='log', resume=resume)
    base_names = ['Epoch']
    metrics = ['Loss', 'Acc']
    logger.set_names(base_names + metrics)

<<<<<<< Updated upstream
    for epoch in range(0, 1000, 10):
        net_ = get_ensemble(epoch, loaders, num_splits=num_splits, width=width, device=device)
        # acc, loss = eval_net(net_, loaders, device=device)
        # str_cpl = '\n[%i] Loss: %.4f Acc: %.4f' 
        # print(str_cpl % (epoch, loss, acc))
        # logs = [epoch, loss, acc]
        # logger.append(logs)

        # switch network of rbLog
        print(epoch)
        net.load_state_dict(net_.state_dict())
        rbLog.update(epoch)
=======
    for epoch in range(start_epoch, end_epoch, 10):
        acc, loss = eval_ensemble(loaders, epoch=epoch, device=device, width=width, num_splits=num_splits)
        str_cpl = '\n[%i] Loss: %.4f Acc: %.4f'
        print(str_cpl % (epoch, loss, acc))
        logs = [epoch, loss, acc]
        logger.append(logs)
>>>>>>> Stashed changes

    rbLog.close()
    print('-- Finished.. %.3f mins' % ((time.time() - start) / 60.0))

