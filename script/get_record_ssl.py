#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re

from src.adversary import attack, scale_step
from src.preprocess import get_loaders
from src.analyses import get_net
from src.utils import str2bool
from src.utils import accuracy, Dict2Obj

def save_data(path, path_key, data, config=None):
    path_tmp = path_key
    if config.state != 'last':
        path_tmp += '_%s' % config.state
    if config.loader != 'train':
        path_tmp += '_%s' % config.loader
    path_tmp += '.npy'
    path_tmp= '%s/%s' % (path, path_tmp)
    with open(path_tmp, 'wb') as f:
        np.save(f, data)

def get_record(net, loaders, path='.', config=None):
    if config.loader == 'train':
        loader = loaders.trainloader
    else:
        loader = loaders.testloader

    if config.get_pred:
        preds = []
    if config.get_softmax:
        softmaxs = []
    if config.get_margin:
        margins = []
    for e, (inputs, labels, weights) in enumerate(loader):
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
            acc, = accuracy(outputs.data, labels.data)
            softmax = F.softmax(outputs, dim=1).detach()

            if config.get_pred:
                preds = np.append(preds, outputs.max(1)[1].cpu().numpy())
            if config.get_softmax:
                softmaxs.append(softmax.cpu().numpy())
            if config.get_margin:
                max2 = softmax.topk(2, 1, True, True)[0]
                margins = np.append(margins, (max2[:, 0] - max2[:, 1]).cpu().numpy())

        print('----------- [%i/%i] --- ACC: %.3f ------' % (e, len(loader), acc.item()), end='\r')
    print('-----> end.')
    
    if config.get_pred:
        save_data(path, 'record_pred_last', preds, config=config)
    if config.get_softmax:
        softmaxs = np.vstack(softmaxs)
        save_data(path, 'record_softmax_last', softmaxs, config=config)
    if config.get_margin:
        save_data(path, 'record_margin_last', margins, config=config)


def evaluate():

    # path_for_sed='checkpoints2/sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_idx_label_split_labeled=4000_pseudolabel_iter=20_th=0.9999_type=value_nonratio_func=linear_min=0.9_model=sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_idx_label_split_labeled=4000'
    # path = path_for_sed
    path = 'checkpoints/sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_idx_label_split_labeled=4000'
    model = 'wrn'
    depth = 28
    width = 2
    gpu_id = 4
    loader='test'
    state='last'

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict()
    config['dataset'] = 'cifar10' # 'cifar100' #
    config['loader'] = loader
    config['data_dir'] = '/home/chengyu/Initialization/data'
    config['batch_size'] = 128
    config['traintest'] = False # True
    config['shuffle_train_loader'] = False # if break, maintain loader order when continue
    config['random_augment'] = False
    config['state'] = state
    config['device'] = device
    config['get_pred'] = False
    config['get_softmax'] = True
    config['get_margin'] = False
    config['soft_label'] = False
    config['soft_label_test'] = False
    assert(any([config[get_key] for get_key in ['get_pred', 'get_softmax', 'get_margin']])), 'No get query set.'
    config = Dict2Obj(config)

    print('>>>>>>>>>>> get loader..')
    loaders = get_loaders(dataset=config.dataset,
                          batch_size=config.batch_size,
                          shuffle_train_loader=config.shuffle_train_loader,
                          data_dir=config.data_dir,
                          config=config)

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

    def get_iters(path):
        iters = []
        for filename in os.listdir(path):
            if re.match(r'iteration-\d+', filename):
                iters.append(int(re.findall(r'iteration-(\d+)', filename)[0]))
        return sorted(iters)

    print('>>>>>>>>>>> start evaluating..')
    path = os.path.join('/home/chengyu/Initialization', path)
    # Standard
    print('-----> get net..')
    net = get_net(path,
                  num_classes=loaders.num_classes,
                  n_channel=loaders.n_channel,
                  feature=None,
                  model=model,
                  depth=depth,
                  width=width,
                  state=model_state,
                  device=device)

    print('-----> get predictions..')
    get_record(net, loaders, path, config)

    # # iterations in SSL
    # for iter_ in get_iters(path): # SSL
    #     dirname = 'iteration-%i' % iter_
    #     path_iter = os.path.join(path, dirname)
    #     print('-----> %s' % dirname)
    #     print('-----> get net..')
    #     net = get_net(path_iter,
    #                     num_classes=loaders.num_classes,
    #                     n_channel=loaders.n_channel,
    #                     feature=None,
    #                     model=model,
    #                     depth=depth,
    #                     width=width,
    #                     state=model_state,
    #                     device=device)

    #     print('-----> get predictions..')
    #     get_record(net, loaders, path_iter, config)

    print('>>>>>>>>>>> Done.')

if __name__ == '__main__':
    evaluate()
