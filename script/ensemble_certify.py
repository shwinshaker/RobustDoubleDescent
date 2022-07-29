#!./env python

import torch
import torch.nn as nn
import os

from src.analyses import get_net
from src.adversary import AAAttacker

from epoch_ensemble import OutputAvg

def get_nets(epoch, num_splits=2, device=None):
    nets = []
    for split in range(num_splits):
        # print('Model - %i' % split)
        if split == 0:
            path = 'checkpoints/adam_wrn-28-5_gain=1_0_ad_pgd_10_alpha=1_lr=1e-04_mom=0_9_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000'
        else:
            path = 'checkpoints/adam_wrn-28-5_gain=1_0_ad_pgd_10_alpha=1_lr=1e-04_mom=0_9_pgd_10_sub=mean_rb_id_pgd10_epoch_friend_35000_rand_5000_1'

        net = get_net(path, num_classes=10, n_channel=3,
                      model='wrn', depth=28, width=5, state='model-%i.pt' % epoch,
                      device=device)
        net.eval()
        nets.append(net)
    return nets

gpu_id = 6
epoch = 950
num_splits = 2

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nets = get_nets(epoch, num_splits=num_splits, device=device)
net = OutputAvg(nets[0], nets[1])
net.eval()
assert(not net.training)
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        assert(not m.training)

attacker = AAAttacker(net=net,
                      normalize=True,
                      mode='standard',
                      path='./tmp',
                      log_path='log_certify_best_epoch=%i.txt' % epoch,
                      device=device,
                      data_dir='/home/chengyu/Initialization/data')
attacker.evaluate()

