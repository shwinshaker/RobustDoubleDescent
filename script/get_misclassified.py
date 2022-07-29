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
from src.utils import str2bool
from src.utils import accuracy, Dict2Obj

def save_data(path, path_key, data, config=None):
    path_tmp = path_key + '_%s' % config.state
    if config.adversary is not None:
        path_tmp += '_%s' % config.adversary
    if config.loader != 'test':
        path_tmp += '_%s' % config.loader
    path_tmp += '.npy'
    path_tmp= '%s/%s' % (path, path_tmp)
    with open(path_tmp, 'wb') as f:
        np.save(f, data)

def get_misclassified(net, loaders, path='.', config=None):
    if config.loader == 'train':
        loader = loaders.trainloader
    else:
        loader = loaders.testloader
    criterion = nn.CrossEntropyLoss()

    if config.adversary is not None:
        eps = scale_step(config.eps, config.dataset, device=config.device)
        pgd_alpha = scale_step(config.pgd_alpha, config.dataset, device=config.device)

    mis_ids = []
    if config.get_confidence:
        confidences = []
    if config.get_margin:
        margins = []
    for e, (inputs, labels, weights) in enumerate(loader):
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        net.eval()

        if config.adversary is not None:
            inputs_ad, _ = attack(net,
                                criterion,
                                inputs, labels,
                                adversary=config.adversary,
                                eps=eps,
                                pgd_alpha=pgd_alpha,
                                pgd_iter=10,
                                randomize=True,
                                config=config)
            net.train()
            inputs_ = inputs_ad
        else:
            inputs_ = inputs

        outputs = net(inputs_)
        softmaxs = F.softmax(outputs, dim=1).detach()
        if config.get_confidence:
            confidences = np.append(confidences, softmaxs.max(1)[0].cpu().numpy())
        if config.get_margin:
            max2 = softmaxs.topk(2, 1, True, True)[0]
            margins = np.append(margins, (max2[:, 0] - max2[:, 1]).cpu().numpy())
            # margins.append((max2[:, 0] - max2[:, 1]).cpu().numpy())
        acc, = accuracy(outputs.data, labels.data)

        ids = weights['index'].to(config.device)
        _, preds = outputs.max(1)
        mis_ids = np.append(mis_ids, ids[~preds.squeeze().eq(labels)].cpu().numpy())

        print('----------- [%i/%i] --- # misclassified: %i -- ACC: %.3f ------' % (e, len(loader), len(mis_ids), acc.item()))
    
    save_data(path, 'ids_misclassified', mis_ids, config=config)

    if config.get_confidence:
        save_data(path, 'confidence', confidences, config=config)

    if config.get_margin:
        save_data(path, 'margin', margins, config=config)


def evaluate(model, depth, width, state='best',
         path='.', gpu_id='0',
         adversary='pgd',
         dataset='cifar10', loader='test',
         data_dir='/home/chengyu/Initialization/data'):

    if adversary not in ['pgd', None]:
        raise NotImplementedError()

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict()
    config['eps'] = 8
    config['pgd_alpha'] = 2
    config['dataset'] = dataset
    config['loader'] = loader
    config['data_dir'] = data_dir
    config['batch_size'] = 128
    config['traintest'] = False # True
    config['shuffle_train_loader'] = False # if break, maintain loader order when continue
    config['random_augment'] = True # False # Produce an adversarial counterpart of the original image
    config['adversary'] = adversary
    config['state'] = state
    config['device'] = device
    config['get_confidence'] = True
    config['get_margin'] = True
    config['soft_label'] = False
    config['soft_label_test'] = False
    config = Dict2Obj(config)

    print('>>>>>>>>>>> get loader..')
    loaders = get_loaders(dataset=config.dataset,
                          batch_size=config.batch_size,
                          shuffle_train_loader=config.shuffle_train_loader,
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
    get_misclassified(net, loaders, path, config)

    print('>>>>>>>>>>> Done.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=20, type=int, help='model depth')
    parser.add_argument('--width', default=64, type=int, help='model width')
    parser.add_argument("-p", "--path", type=str, default='.', help="model path")
    parser.add_argument('-d', "--state", default='best', type=str, help='model state')
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    parser.add_argument("-ad", "--adversary", default=None, type=str, help="adversary (or not)")
    parser.add_argument("-l", "--loader", default='test', type=str, help="loader (train or test)")
    parser.add_argument("--dataset", default='cifar10', type=str, help="dataset")
    args = parser.parse_args()

    evaluate(model=args.model, depth=args.depth, width=args.width, state=args.state,
        path=args.path, gpu_id=args.gpu,
        adversary=args.adversary, dataset=args.dataset, loader=args.loader)
