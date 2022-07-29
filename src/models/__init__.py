#!./env python

from .vgg import *
from .wideresnet import *
from .preactresnet import *

def get_net(config, loaders):
    if 'vgg' in config.model:
        net = globals()[config.model](batch_norm=config.bn, num_classes=loaders.num_classes, n_channel=loaders.n_channel).to(config.device)

    elif 'wrn' in config.model:
        net = globals()[config.model](depth=config.depth, widen_factor=config.width, num_classes=loaders.num_classes, n_channel=loaders.n_channel).to(config.device)

    elif config.model in ['PreActResNet18']:
        net = globals()[config.model](num_classes=loaders.num_classes, n_channel=loaders.n_channel).to(config.device)

    else:
        raise KeyError(config.model)

    return net
