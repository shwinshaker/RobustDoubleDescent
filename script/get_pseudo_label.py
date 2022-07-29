
import torch
import torch.nn.functional as F
import numpy as np
import os

from src.preprocess import get_loaders
from src.analyses import get_net
from src.utils import accuracy, Dict2Obj

def get_pseudo_labels(net, loader, config=Dict2Obj({'device': None})):
    ids_list = []
    preds_list = []
    confs_list = []
    labels_list = []
    mis_ids_list = []
    
    n_correct = 0
    n_total = 0
    for e, (inputs, labels, weights) in enumerate(loader):
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
            acc, = accuracy(outputs.data, labels.data)
        
            preds = outputs.max(1)[1]
            preds_list.append(preds.cpu().numpy())

            softmaxs = F.softmax(outputs, dim=1).detach()
            confs_list.append(softmaxs.max(1)[0].cpu().numpy())

            ids = weights['index'].to(config.device)
            ids_list.append(ids.cpu().numpy())
            mis_ids_list.append(ids[~preds.squeeze().eq(labels)].cpu().numpy())
            labels_list.append(labels.cpu())
        
            n_correct += (preds.squeeze().eq(labels)).sum()
            n_total += inputs.size(0)
            
        # print('----------- [%i/%i] --- # correct: %i -- Acc (Batch): %.3f ------' % \
            #   (e, len(loader), n_correct, acc.item()), end='\r')

    print()
    print('----------- # correct: %i -- Acc: %.3f -- Noise: %.3f ---' % (n_correct, n_correct/n_total, 1-n_correct/n_total))
    
    results = {'pseudo_label': np.hstack(preds_list),
               'true_label': np.hstack(labels_list),
               'confidence': np.hstack(confs_list),
               'index': np.hstack(ids_list),
               'mis_index': np.hstack(mis_ids_list),
              }    
    return results


def pseudo_label(config, dataset, path_unlabeled_idx, model_path, model_state, gpu_id, save_dir='.'):

    print('---- Set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('---- Get unlabeled dataloader..')
    # Do we need random augment when pseudo-labeling?
    # -- Not used in `In defense of pseudo-labeling...`
    print(' --- dataset: %s' % dataset)
    print(' --- path: %s' % path_unlabeled_idx)
    with open(path_unlabeled_idx, 'rb') as f:
        trainsubids = np.load(f)
    loaders_unlabeled = get_loaders(dataset=dataset,
                                    trainsubids=trainsubids,
                                    shuffle_train_loader=False, 
                                    data_dir='/home/chengyu/Initialization/data',
                                    config=Dict2Obj({'random_augment': False}))

    print('---- Get model..')
    print(' --- model path: %s' % model_path)
    print(' --- model: %s  depth: %i  width: %i' % (config.model, config.depth, config.width))
    print(' --- model state: %s' % model_state)
    net = get_net(config,
                  model_path,
                  num_classes=loaders_unlabeled.num_classes,
                  n_channel=loaders_unlabeled.n_channel,
                  state=model_state,
                  device=device)

    print('---- Get predictions..')
    results = get_pseudo_labels(net, loaders_unlabeled.trainloader, config=Dict2Obj({'device': device}))
    assert(np.all(results['true_label'] == np.array(loaders_unlabeled.trainset.targets)))

    print('---- Save pseudo labels..')
    print(' --- save path: %s' % save_dir)
    results['model_path'] = os.path.split(model_path)[1]
    results['model_state'] = model_state
    results['path_unlabeled_idx'] = path_unlabeled_idx
    torch.save(results, '%s/pseudo_state=%s_unlabeled=%s.pt' % (save_dir,
                                                                model_state.rstrip('.pt'),
                                                                os.path.split(path_unlabeled_idx)[1].rstrip('.npy')))


if __name__ == '__main__':

    dataset = 'cifar10' # 'cifar100' # 
    gpu_id = 6

    config = Dict2Obj({
        'model': 'wrn', # 'vit', # 'ffn',
        'bn': True,
        'depth': 28,
        'width': 1, # 3072,
        'dataset': dataset,
        # 'vit_heads': 6,
        # 'vit_patch_size': 16,
        # 'ffn_act': 'tanh',
    })

    model_state = 'model.pt'
    if dataset == 'cifar10':
        # model_path = 'checkpoints/sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_idx_label_split_labeled=4000'
        # path_unlabeled_idx = 'data_subsets/ssl_idx_label_split_unlabeled=46000.npy'
        # model_path = 'checkpoints_fifth/sgd_ffn-1-3072_bn_lr=5e-02_bs=64_wd=0_0005_mom=0_9_crl=1_softmax_linear_ntrain=40'
        # model_path = 'checkpoints_fifth/sgd_vit1-1_patch=4_lr=5e-03_bs=64_wd=0_0005_mom=0_9_crl=1_softmax_linear_ntrain=40'
        # model_path = 'checkpoints_fifth/sgd_vit1-1_lr=5e-03_bs=64_wd=0_0005_mom=0_9_crl=1_softmax_linear_ntrain=40-2'
        model_path = 'checkpoints_fifth/sgd_wrn-28-1_lr=5e-02_bs=64_wd=0_0005_mom=0_9_ntrain=200-1'
        # model_path = 'checkpoints_fifth/sgd_wrn-28-1_lr=5e-02_bs=64_wd=0_0005_mom=0_9_crl=1_softmax_linear_ntrain=200-1'
        path_unlabeled_idx = '%s/id_unlabeled_cifar10_size=49800.npy' % model_path
    elif dataset == 'cifar100':
        path_unlabeled_idx = 'data_subsets/ssl_cifar100_idx_label_split_unlabeled=46000.npy'
        model_path = 'checkpoints/cifar100_sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_cifar100_idx_label_split_labeled=4000'


    pseudo_label(config, dataset, path_unlabeled_idx, model_path, model_state, gpu_id=gpu_id, save_dir=model_path)

    


