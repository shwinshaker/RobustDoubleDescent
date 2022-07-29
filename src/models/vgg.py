#!./env python

import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, out=512):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(out, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, n_channel=3, batch_norm=False):
    layers = []
    in_channels = n_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': {'arch': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'out': 512},
    'B': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'out': 512},
    'D': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'out': 512},
    'E': {'arch': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 'out': 512},
}

def vgg11(**kwargs):
    c = 'A'
    model = VGG(make_layers(cfg[c]['arch'],
                            n_channel=kwargs['n_channel'],
                            batch_norm=kwargs['batch_norm']),
                out=cfg[c]['out'],
                num_classes=kwargs['num_classes'])
    return model

def vgg13(**kwargs):
    c = 'B'
    model = VGG(make_layers(cfg[c]['arch'],
                            n_channel=kwargs['n_channel'],
                            batch_norm=kwargs['batch_norm']),
                out=cfg[c]['out'],
                num_classes=kwargs['num_classes'])
    return model

def vgg16(**kwargs):
    c = 'D'
    model = VGG(make_layers(cfg[c]['arch'],
                            n_channel=kwargs['n_channel'],
                            batch_norm=kwargs['batch_norm']),
                out=cfg[c]['out'],
                num_classes=kwargs['num_classes'])
    return model

def vgg19(**kwargs):
    c = 'E'
    model = VGG(make_layers(cfg[c]['arch'],
                            n_channel=kwargs['n_channel'],
                            batch_norm=kwargs['batch_norm']),
                out=cfg[c]['out'],
                num_classes=kwargs['num_classes'])
    return model
