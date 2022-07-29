#!./env python

from src.preprocess import get_loaders
from src.analyses import get_net, get_ad_examples
from src.utils import Dict2Obj
from src.utils import DeNormalizer
from src.preprocess.dataset_info import dataset_stats
import numpy as np
import torch

import os

gpu_id = 5
adversary = 'pgd'
suffix = '' # 'epoch=10'
eps = 12
pgd_alpha = 2
phase = 'train'
continue_at = 0

## Set environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Get data
config = dict()
config['dataset'] = 'cifar10'
config['data_dir'] = '/home/chengyu/Initialization/data'
config['traintest'] = True
config['shuffle_train_loader'] = False # if break, maintain loader order when continue
config['random_augment'] = False # Produce an adversarial counterpart of the original image
config['model'] = 'PreActResNet18'
config['device'] = device
# config['path'] = 'checkpoints1/sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
config['path'] = 'checkpoints1/sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10-0'
# config['path'] = 'checkpoints1/sgd_PreActResNet18_gain=1_0_wd=0_0005_mom=0_9_pgd_10'
config['model_state'] = 'best_model.pt'
# config['model_state'] = 'model-10.pt'
config = Dict2Obj(config)

# labelnoisyids = []
# if hasattr(config, 'noise_subset_path') and config.noise_subset_path:
#     with open(config.noise_subset_path, 'rb') as f:
#         labelnoisyids = np.load(f)

loaders = get_loaders(dataset=config.dataset,
                      shuffle_train_loader=config.shuffle_train_loader,
                      data_dir=config.data_dir,
                      config=config)

## Get net
if eps > 0:
    net_best = get_net(config, config.path, num_classes=10, n_channel=loaders.n_channel, device=device, state=config.model_state)

## Get aa
if phase == 'train':
    loader = loaders.trainloader
else:
    loader = loaders.testloader

def push_pt(inputs, targets, indices, file_name='ad_tmp.pt'):
    inputs = inputs.detach().cpu()
    targets = targets.detach().cpu()
    indices = indices.detach().cpu()
    if len(inputs.size()) == 3:
        inputs = inputs.unsqueeze(0)

    if os.path.exists(file_name):
        records = torch.load(file_name)
        records['inputs'] = torch.cat([records['inputs'], inputs])
        records['targets'] = torch.cat([records['targets'], targets])
        records['indices'] = torch.cat([records['indices'], indices])
    else:
        records = dict()
        records['inputs'] = inputs
        records['targets'] = targets
        records['indices'] = indices
    torch.save(records, file_name)

base_name = '%s_set' % phase
if eps > 0:
    base_name += '_eps=%i' % eps
if suffix:
    file_name = 'data/%s_%s_%s.pt' % (adversary, base_name, suffix)
else:
    file_name = 'data/%s_%s.pt' % (adversary, base_name)

denormalize = DeNormalizer(dataset_stats[config.dataset]['mean'],
                           dataset_stats[config.dataset]['std'],
                           loaders.n_channel, device)

for e, (inputs, labels, weights) in enumerate(loader):
    if continue_at > 0 and e < continue_at:
        print('----------- [%i/%i - Done] ---------' % (e, len(loader)))
        continue
    else:
        print('----------- [%i/%i] ---------' % (e, len(loader)))

    inputs, labels = inputs.to(device), labels.to(device)
    if eps > 0:
        if adversary == 'aa':
            inputs, _ = get_ad_examples(net_best, inputs, labels, eps=eps,
                                        dataset='cifar10', adversary='aa', device=device,
                                        log_path='log_certify_%s.txt' % base_name, path='./tmp')
        elif adversary == 'pgd':
            inputs, _ = get_ad_examples(net_best, inputs, labels, eps=eps,
                                        dataset='cifar10', adversary='pgd',
                                        pgd_iter=10, pgd_alpha=pgd_alpha, randomize=True,
                                        device=device)
        else:
            raise KeyError(adversary)
    inputs = denormalize(inputs) # denormalize 
    push_pt(inputs, labels, weights['index'], file_name=file_name)

    # if e > 0:
    #     break

