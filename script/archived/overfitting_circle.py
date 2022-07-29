#!./env python

import numpy as np
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils import str2bool

def sample_spherical(n_samples=1000, n_dim=3):
    vec = np.random.randn(n_samples, n_dim)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def rand_shuffle(X, y):
    stack = np.hstack([X, y[:, None]])
    np.random.shuffle(stack)
    return stack[:, :-1], stack[:, -1]
    
def get_circle(n_samples=1000, n_dim=2, noise=0.1, factor=0.8, shuffle=True, random_state=None):
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
        
    generator = check_random_state(random_state)
    outer_circ = sample_spherical(n_samples=n_samples_out, n_dim=n_dim)
    inner_circ = sample_spherical(n_samples=n_samples_in, n_dim=n_dim) * factor

    X = np.vstack([outer_circ, inner_circ])
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])
    
    if shuffle:
        X, y = rand_shuffle(X, y)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(outputs, labels):
    assert(outputs.size(0) == labels.size(0))
    assert(labels.size(1) == 1) 
    return ((outputs > 0) == labels).sum() * 1.0 / labels.size(0)    
    
    
class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], np.array([self.y[idx]])
     
        
def fgsm(net, criterion, X, y, eps=0.1):

    net.eval()
    net.zero_grad()
    
    delta = torch.zeros_like(X, requires_grad=True)
    loss = criterion(net(X + delta), y)
    loss.backward()
    
    return X + eps * delta.grad.detach().sign()


def test(net, criterion, loader, verbose=False, device='cpu'):
    
    net.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (inputs, labels) in enumerate(loader, 0):
        inputs, labels = inputs.float(), labels.float()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs.data, labels.data)

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc.item(), inputs.size(0))
        
        if verbose:
            print('%.4f - %.4f' % (losses.avg, accs.avg), end='\r')

    net.train()
    return losses.avg, accs.avg


def ad_test(net, criterion, loader, eps=0.1, verbose=False, device='cpu'):
    
    net.eval()
    net.zero_grad()
    
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (inputs, labels) in enumerate(loader, 0):
        inputs, labels = inputs.float(), labels.float()
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_ad = fgsm(net, criterion, inputs, labels, eps=eps)
        
        diff = (inputs_ad - inputs).abs().max().item()
        assert(np.isclose(diff, eps, rtol=1e-03, atol=1e-05)), (diff, eps)

        outputs_ad = net(inputs_ad)
        loss_ad = criterion(outputs_ad, labels)
        acc_ad = accuracy(outputs_ad.data, labels.data)

        losses.update(loss_ad.item(), inputs.size(0))
        accs.update(acc_ad.item(), inputs.size(0))
        
        if verbose:
            print('%.4f - %.4f' % (losses.avg, accs.avg), end='\r')

    net.train()
    return losses.avg, accs.avg
        
    
def train(net, criterion, optimizer, loaders, epochs=1000, eps=0.1, adversary_test=False, device='cpu', print_interval=10):
    
    stats = {'train_loss': [],
             'train_acc': [],
             'train_loss_ad': [],
             'train_acc_ad': [],
             'test_loss': [],
             'test_acc': [],
             'test_loss_ad': [],
             'test_acc_ad': [],
            }
    
    losses = AverageMeter()
    accs = AverageMeter()
        
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(loaders['train']):
            inputs, labels = inputs.float(), labels.float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)# .mean()
            acc = accuracy(outputs.data, labels.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), inputs.size(0))
            accs.update(acc.item(), inputs.size(0))
    
        stats['train_loss'].append(losses.avg)
        stats['train_acc'].append(accs.avg) 
        if adversary_test:
            loss_ad, acc_ad = ad_test(net, criterion, loaders['train'], eps=eps, device=device)
            stats['train_loss_ad'].append(loss_ad)
            stats['train_acc_ad'].append(acc_ad)
            
        ## Test
        loss, acc = test(net, criterion, loaders['test'], device=device)
        stats['test_loss'].append(loss)
        stats['test_acc'].append(acc)
        acc_ad = -1
        if adversary_test:
            loss_ad, acc_ad = ad_test(net, criterion, loaders['test'], eps=eps, device=device)
            stats['test_loss_ad'].append(loss_ad)
            stats['test_acc_ad'].append(acc_ad)
        
        if epoch % print_interval == 0:
            # print('[%i/%i] Train Acc: %.4f - Test Acc: %.4f' % (epoch, epochs, accs.avg, acc), end='\r')
            print('[%i/%i] Train Acc: %.4f - Test Acc: %.4f' % (epoch, epochs, accs.avg, acc))
        # print('[%i] Test Acc: %.4f - Test Acc Ad: %.4f' % (epoch, acc, acc_ad), end='\r')

        losses.reset()
        accs.reset()
        
    return stats


class DLN(nn.Module):
    """
        Deep linear network, for illustration
    """

    def __init__(self, features=[], dim0=3, n_class=1):
        super(DLN, self).__init__()
        self.features = nn.Sequential()
        
        dim = dim0
        for idx, l in enumerate(features):
            layer = nn.Sequential()
            if l['width']:
                layer.add_module('linear', nn.Linear(dim, l['width']))
                dim = l['width']
            if l['act']:
                layer.add_module(l['act'], self.__get_activation(l['act']))
            if l['bn']:
                layer.add_module('bn', nn.BatchNorm1d(dim, affine=True))
            self.features.add_module('layer%i' % idx, layer)
        # self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(dim, n_class)
        self.activation = nn.Sigmoid()
        
        # self.__initialize()

    def __get_activation(self, act):
        if act == 'tanh':
            return nn.Tanh()
        if act == 'relu':
            return nn.ReLU()
        raise KeyError(act)
        
        
    def __initialize(self):      
        for m in self.features:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                # nn.init.xavier_normal_(m.weight.data)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.orthogonal_(m.weight.data)
                m.bias.data.zero_()
        # nn.init.xavier_uniform_(self.classifier.weight.data)
        # nn.init.xavier_normal_(self.classifier.weight.data)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.orthogonal_(self.classifier.weight.data)
        self.classifier.bias.data.zero_()
        
    def forward(self, x):
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.activation(x)
        return x # torch.sigmoid(x) 
    
def make_layer(width=[], depth=5, act='tanh', bn=False):
    if not depth or not width:
        # logisitc
        return [{'width': None, 'act': None, 'bn': bn}]
#     if not width:
#         return [{'width': None, 'act': act, 'bn': bn}]
    if isinstance(width, int):
        return [{'width': width, 'act': act, 'bn': bn}] * depth
    elif isinstance(width, list):
        return [{'width': w, 'act': act, 'bn': bn} for w in width]
    raise TypeError(width)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', default=10000, type=int, help='# samples (half train, half test)')
    parser.add_argument('--factor', default=0.9, type=float, help='scaling factor of the dataset')
    parser.add_argument('--epochs', default=10000, type=int, help='epochs')
    parser.add_argument('--print-interval', default=100, type=int, help='# epoch to print stats')
    parser.add_argument('--depth', default=2, type=int, help='model depth')
    parser.add_argument('--width', default=10, type=int, help='model width')
    parser.add_argument("--create-dataset", type=str2bool, nargs='?', const=True, default=False, help="create dataset?")
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    args = parser.parse_args()


    ## Parameters
    gpu_id = args.gpu
    create_dataset = args.create_dataset
    n_samples = args.nsamples
    noise = 0.1 # fix noise
    factor = args.factor # vary factor
    dim = 5

    # -- training paras
    epochs = args.epochs
    print_interval = args.print_interval
    lr = 0.001 # default learning rate for adam
    depth = args.depth
    width = args.width
    act = 'tanh'

    ## Set environment
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ## Get dataset
    if create_dataset:
        data, labels = get_circle(n_samples=n_samples, n_dim=dim, factor=factor, noise=noise, shuffle=True)
        trainsize = len(data) // 2  # int(len(data) / 10 * 9)
        trainset = BinaryDataset(data[:trainsize], labels[:trainsize])
        testset = BinaryDataset(data[trainsize:], labels[trainsize:])
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = DataLoader(testset, batch_size=128, shuffle=False)
        loaders = {'train': trainloader, 'test': testloader}
        torch.save(loaders, 'tmp/circle/data-factor=%g.pt' % factor)
    else:
        loaders = torch.load('tmp/circle/data-factor=%g.pt' % factor)

    ## Train
    criterion = nn.BCEWithLogitsLoss()
    print('---------- factor=%g ------- width=%i ----------------' % (factor, width))
    
    net = DLN(make_layer(width=width, depth=depth, bn=True), dim0=dim, n_class=1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    stats = train(net, criterion, optimizer, loaders, epochs=epochs, adversary_test=False, device=device, print_interval=print_interval)
    
    torch.save(stats, 'tmp/circle/stats-factor=%g-width=%i.pt' % (factor, width))





