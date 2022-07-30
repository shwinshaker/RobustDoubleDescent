#!./env python
import torch
from collections import OrderedDict
import os

path = 'checkpoints/tiny-imagenet_sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
checkpoint = 'best_model_parallel.pt'
save_checkpoint = 'best_model.pt'

# original saved file with DataParallel
state_dict = torch.load(os.path.join(path, checkpoint))
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
torch.save(new_state_dict, os.path.join(path, save_checkpoint))
