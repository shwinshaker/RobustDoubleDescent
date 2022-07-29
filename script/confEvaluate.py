#!./env python

import torch
import numpy as np
import os
import argparse

import torch.nn as nn
import torch.nn.functional as F

from src.preprocess import get_loaders
from src.analyses import get_net
from src.utils import Dict2Obj, AverageMeter
from src.utils import mse_one_hot, Confidence


class ConfEvaluator:

    __available_metrics = ['NLL', 'Brier', 'ECE', 'AURC', 'TV', 'KL'] # TV and KL require true label distribution

    def __init__(self, net, loaders, metrics, save_path, config):
        assert(all([m.split('-')[0] in self.__available_metrics for m in metrics]))
        self.config = config
        self.device = config.device

        self.net = net
        self.loader = loaders.testloader
        self.metrics = metrics
        self.save_path = save_path

        self.conf = Confidence(metrics=metrics, num_classes=loaders.num_classes, device=config.device)

    def eval_confidence(self):
        results = self.conf.evaluate(self.net, self.loader)
        print_string = ' -- '.join(['%s: %.4f' % (metric, results[metric]) for metric in results])
        print('----------- %s ----------' % (print_string))
        torch.save(results, self.save_path)
        return results


def conf_evaluate(model, depth, width, dataset='cifar10',
                  path='.', save_path=None, state='last', gpu_id='0',
                  metrics=['NLL', 'Brier']):

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('>>>>>>>>>>> set config..')
    if save_path is None:
        save_path = 'log'
        save_path += '_confidence_%s' % state
        save_path += '_' + '-'.join(metrics)
        save_path += '.pt'
        save_path = os.path.join(path, save_path)

    config = {'dataset': dataset,
              'soft_label': False,
              'soft_label_test': False,
              'data_dir': '/home/chengyu/Initialization/data',
              'shuffle_train_loader': False, # if break, maintain loader order when continue
              'random_augment': False, # Produce an adversarial counterpart of the original image
              'batch_size': 1000, # 128
              'traintest': False, # True
              'device': device,
            }
    config = Dict2Obj(config)


    print('>>>>>>>>>>> get loader..')
    loaders = get_loaders(dataset=config.dataset,
                          data_dir=config.data_dir,
                          shuffle_train_loader=config.shuffle_train_loader,
                          config=config)


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
    net = get_net(path,
                  num_classes=loaders.num_classes,
                  n_channel=loaders.n_channel,
                  feature=None,
                  model=model,
                  depth=depth,
                  width=width,
                  state=model_state,
                  device=device)


    print('>>>>>>>>>>> start evaluating..')
    evaluator = ConfEvaluator(net, loaders, metrics, save_path, config)
    evaluator.eval_confidence()

    print('>>>>>>>>>>> Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=20, type=int, help='model depth')
    parser.add_argument('--width', default=64, type=int, help='model width')
    parser.add_argument("-p", "--path", type=str, help="model path")
    parser.add_argument("-sp", "--save_path", type=str, help="save path")
    parser.add_argument('-d', "--state", default='last', type=str, help='model state')
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    parser.add_argument("--dataset", default='cifar10', type=str, help="dataset")
    args = parser.parse_args()

    # metrics = ['NLL', 'Brier', 'TV', 'KL', 'ECE', 'AURC']
    metrics = ['NLL', 'Brier', 'ECE', 'AURC']
    # metrics = ['ECE-%i' % i for i in range(10, 110, 20)]

    conf_evaluate(model=args.model, depth=args.depth, width=args.width,
                  path=args.path, save_path=args.save_path, state=args.state, gpu_id=args.gpu,
                  metrics=metrics, dataset=args.dataset)


