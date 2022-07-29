#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

from src.preprocess import get_loaders
from src.analyses import get_net
from src.utils import str2bool
from src.utils import accuracy, Dict2Obj

class RobustTracker:

    available_metrics = ['CW', 'FGSM', 'CLEVER']

    def __init__(self, net, loaders, criterion, config):
        assert(all([m in self.available_metrics for m in config.rbTrack]))
        self.metrics = config.rbTrack
        self.device = config.device

        from src.preprocess import dataset_stats
        mean = np.array(dataset_stats[config.dataset]['mean'])
        std = np.array(dataset_stats[config.dataset]['std'])

        from src.utils import DeNormalizer
        self.denormalize = DeNormalizer(mean, std, loaders.n_channel, config.device)

        # create an ART instance wrapper
        from art.estimators.classification import PyTorchClassifier
        self.classifier = PyTorchClassifier(model=net,
                                            loss=criterion,
                                            input_shape=tuple(loaders.shape),
                                            nb_classes=loaders.num_classes,
                                            channels_first=True,
                                            preprocessing=(mean.reshape(loaders.n_channel, 1, 1),
                                                           std.reshape(loaders.n_channel, 1, 1)),
                                            clip_values=(0, 1),
                                            device_type='gpu')

        # create methods
        self.methods = dict()
        if 'FGSM' in self.metrics:
            from art.attacks.evasion import FastGradientMethod
            self.FGSM = FastGradientMethod(self.classifier, norm=np.inf, targeted=False, eps_step=config.eps_step/255, minimal=True)
            self.methods['FGSM'] = self.__get_FGSM
        if 'CW' in self.metrics:
            from art.attacks.evasion import CarliniLInfMethod
            self.CW = CarliniLInfMethod(self.classifier, targeted=False, confidence=0)
            self.methods['CW'] = self.__get_CW
        if 'CLEVER' in self.metrics:
            from art.metrics import clever_u
            self.methods['CLEVER'] = self.__get_CLEVER

    def get_distance(self, inputs):
        perturbs = dict([(n, []) for n in self.metrics])
        inputs = inputs.to(self.device)
        inputs = self.__to_art(inputs)
        for name in self.methods:
            perturbs[name].extend(self.methods[name](inputs))
        return perturbs

    def __get_FGSM(self, inputs):
        inputs_ad = self.__from_art(self.FGSM.generate(inputs))
        # TODO: output label
        return self.__distance(inputs, inputs_ad)

    def __get_CW(self, inputs):
        inputs_ad = self.__from_art(self.CW.generate(inputs, verbose=False))
        return self.__distance(inputs, inputs_ad)

    def __get_CLEVER(self, inputs):
        li = []
        for i, inputt in enumerate(inputs):
            li.append(clever_u(self.classifier, inputt.numpy(), nb_batches=2, batch_size=10, radius=0.3, norm=np.inf))
        return li
    
    def __distance(self, inputs, inputs_ad):
        """
            l-inf
        """
        inputs = inputs.view(inputs.size(0), -1)
        inputs_ad = inputs_ad.view(inputs_ad.size(0), -1)
        return torch.norm(inputs_ad - inputs, float('inf'), dim=1).tolist()

    def __to_art(self, tensor):
        # from torch type (cuda tensor) to art type (cpu tensor)
        return self.denormalize(tensor).cpu().detach()

    def __from_art(self, array):
        # from art type (cpu tensor) to torch type (cuda tensor)
        return torch.Tensor(array)

    def close(self):
        self.logger.close()


def save_data(path, path_key, data, config=None):
    path_tmp = path_key + '_%s' % config.state
    if config.loader != 'test':
        path_tmp += '_%s' % config.loader
    if config.eps_step != 1:
        path_tmp += '_stepsize=%g' % config.eps_step
    path_tmp += '.npy'
    path_tmp= '%s/%s' % (path, path_tmp)
    with open(path_tmp, 'wb') as f:
        np.save(f, data)


def get_perturb(net, loaders, path, save_path, config):
    if config.loader == 'train':
        loader = loaders.trainloader
    else:
        loader = loaders.testloader
    criterion = nn.CrossEntropyLoss()
    tracker = RobustTracker(net, loaders, criterion, config)

    min_perturbs = []
    # dists = np.zeros(len(loaders.trainset))
    for i, (inputs, _, weights) in enumerate(loader):
        inputs = inputs.to(config.device)
        perturb = tracker.get_distance(inputs=inputs)['FGSM']
        # indices = weights['index'].numpy()
        # assert(np.all(min_perturbs[indices] == 0))
        min_perturbs = np.append(min_perturbs, perturb)
        print('---------- [%i/%i] --------- Min: %.5f Max: %.5f' % (i, len(loader), np.min(perturb), np.max(perturb)))
    save_data(path, 'min_perturb', min_perturbs, config=config)


def main(model, depth, width, state='best',
         path='.', save_path=None, gpu_id='0',
         dataset='cifar10', loader='test',
         data_dir='/home/chengyu/Initialization/data'):

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict()
    config['rbTrack'] = ['FGSM']
    config['eps_step'] = 0.1
    config['dataset'] = dataset
    config['loader'] = loader
    config['data_dir'] = data_dir
    config['batch_size'] = 128
    config['traintest'] = False # True
    config['shuffle_train_loader'] = False # if break, maintain loader order when continue
    config['random_augment'] = True # False # Produce an adversarial counterpart of the original image
    config['state'] = state
    config['device'] = device
    config = Dict2Obj(config)

    print('>>>>>>>>>>> get loader..')
    loaders = get_loaders(dataset=config.dataset,
                          batch_size=config.batch_size,
                          shuffle_train_loader=config.shuffle_train_loader,
                          random_augment=config.random_augment,
                          data_dir=config.data_dir,
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
    get_perturb(net, loaders, path, save_path, config)

    print('>>>>>>>>>>> Done.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=20, type=int, help='model depth')
    parser.add_argument('--width', default=64, type=int, help='model width')
    parser.add_argument("-p", "--path", type=str, default='.', help="model path")
    parser.add_argument("-sp", "--save_path", type=str, default=None, help="save path")
    parser.add_argument('-d', "--state", default='best', type=str, help='model state')
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    parser.add_argument("-l", "--loader", default='test', type=str, help="loader (train or test)")
    parser.add_argument("--dataset", default='cifar10', type=str, help="dataset")
    args = parser.parse_args()

    main(model=args.model, depth=args.depth, width=args.width, state=args.state,
         path=args.path, save_path=args.save_path, gpu_id=args.gpu,
         dataset=args.dataset, loader=args.loader)
