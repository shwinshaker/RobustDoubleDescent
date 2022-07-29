import torch
# from collections import OrderedDict
import os
import yaml
from src.utils import Dict2Obj
from src.models import get_net

if __name__ == '__main__':
    # get config file
    path = 'checkpoints/cifar100_sgd_wrn-40-4_lr=5e-02_bs=64_wd=0_0005_mom=0_9'
    src_state = 'model.pt'
    tar_state = 'model.pt'
    
    print(' - path: %s' % path)

    # load net
    print(' -- Loading model..')
    with open('%s/config.yaml' % path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Dict2Obj(config)
    config.device = 'cuda'
    loaders = {
        'num_classes': 100,
        'n_channel': 3,
    }
    net = get_net(config, Dict2Obj(loaders))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(path, src_state), map_location=config.device))

    print(' -- wrapping %s to %s' % (src_state, tar_state))
    torch.save(net.module.state_dict(), os.path.join(path, tar_state))

