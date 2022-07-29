#!./env python

import argparse
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import shutil
import yaml
import json
import torch

from src.utils import check_path
from fractions import Fraction

def check_num(num):
    if type(num) in [float, int]:
        return num

    if isinstance(num, str):
        return float(Fraction(num))

    raise TypeError(num)


def read_config(config_file='config.yaml'):

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -- hyperparas massage --
    # TODO: tracker not implemented for resnet
    if ('loss' in config) and (config['loss'] != 'ce') and (config['adversary'] in ['trades', 'mart', 'fat', 'gairat']):
        raise NotImplementedError('Square loss might not be appropriate for trades like adversary!')

    if 'exTrackOptions' not in config:
        # adapt to old config version
        config['exTrackOptions'] = ['count_wrong', 'epoch_first', 'record_correct', 'count_iters']

    if 'resnet' in config['model']:
        config['paraTrack'] = False
        config['lrTrack'] = False
        config['lipTrack'] = False

    if 'resnet' in config['model']:
        if not config['bn']:
            config['model'] = '%s_fixup' % config['model']

    for key in ['eps', 'eps_test', 'lr', 'wd', 'momentum', 'gamma', 'lmr', 'alpha',
                'label_smoothing', 'loss_flooding']:
        if key in config and config[key] is not None:
            config[key] = check_num(config[key])

    if config['state_path']:
        # append absolute path
        config['state_path'] = os.path.join(os.getcwd(), 'checkpoints', config['state_path'])


    # -- checkpoint set --
    config['checkpoint'] = config['opt']
    if config['dataset'] != 'cifar10':
        config['checkpoint'] = '%s_' % config['dataset'] + config['checkpoint']
    if 'soft_label' in config:
        if config['soft_label']:
            config['checkpoint'] += '_softlabel'
    else:
        config['soft_label'] = False
    if 'aux_data' in config and config['aux_data']:
        config['checkpoint'] += '_auxdata'
    if 'loss' in config and config['loss'] != 'ce':
        config['checkpoint'] += '_%s' % config['loss']
    config['checkpoint'] += '_%s' % config['model']
    if 'ffn' in config['model']:
        config['checkpoint'] += '_%i_%i' % (config['depth'], config['width'])
    if 'resnet' in config['model']:
        config['checkpoint'] += '%i' % config['depth']
        if config['width'] != 16:
            config['checkpoint'] += '_width=%i' % config['width']
    elif config['model'] in ['ResNet18', 'PreActResNet18', 'FixupPreActResNet18', 'PreActResNetGN18']:
        pass
    elif 'wrn' in config['model']:
        if config['width'] < 1:
            config['checkpoint'] += '-%i-%g' % (config['depth'], config['width'])
        else:
            config['checkpoint'] += '-%i-%i' % (config['depth'], config['width'])
    else:
        if config['bn']:
            config['checkpoint'] += '_bn'

    ## --------- regularization
    if 'swa' in config and config['swa']:
        config['checkpoint'] += '_swa_at_%i' % config['swa_start']

    if 'kd' in config and config['kd']:
        config['checkpoint'] += '_kd'
        config['checkpoint'] += '_T=%g' % config['kd_temperature']
        if config['kd_teacher_st']:
            config['checkpoint'] += '_st=%g' % config['kd_coeff_st']
            if isinstance(config['kd_teacher_st'], list):
                config['checkpoint'] += '_ensemble=%i' % len(config['kd_teacher_st'])
                config['kd_teacher_st'] = [os.path.join(os.getcwd(), 'checkpoints', p) for p in config['kd_teacher_st']]
            else:
                config['kd_teacher_st'] = os.path.join(os.getcwd(), 'checkpoints', config['kd_teacher_st'])
        if config['kd_teacher_rb']:
            config['checkpoint'] += '_rb=%g' % config['kd_coeff_rb']
            if isinstance(config['kd_teacher_rb'], list):
                config['checkpoint'] += '_ensemble=%i' % len(config['kd_teacher_rb'])
                config['kd_teacher_rb'] = [os.path.join(os.getcwd(), 'checkpoints', p) for p in config['kd_teacher_rb']]
            else:
                config['kd_teacher_rb'] = os.path.join(os.getcwd(), 'checkpoints', config['kd_teacher_rb'])

    if 'sa' in config and config['sa']:
        if config['sa'] == 'output':
            config['checkpoint'] += '_sa'
        elif config['sa'] == 'swa':
            config['checkpoint'] += '_sa_swa'
            config['swa'] = True
            config['swa_start'] = config['sa_start_at']
            config['swa_interval'] = 1
        else:
            raise KeyError(config['sa'])
        if 'sa_weighted' in config and config['sa_weighted']:
            config['checkpoint'] += '_weighted'
        config['checkpoint'] += '_T=%g' % config['sa_temperature']
        config['checkpoint'] += '_coeff=%g' % config['sa_coeff']
        config['checkpoint'] += '_start_at=%i' % config['sa_start_at']

    if 'bootstrap' in config and config['bootstrap']:
        config['checkpoint'] += '_bootstrap_%s' % config['bootstrap']
        if config['bootstrap'] == 'swa':
            config['swa'] = True
            config['swa_start'] = config['bootstrap_start_at']
            config['swa_interval'] = 1
        config['checkpoint'] += '_%s' % config['bootstrap_type']
        config['checkpoint'] += '_T=%g' % config['bootstrap_temperature']
        config['checkpoint'] += '_coeff=%g' % config['bootstrap_coeff']
        if 'bootstrap_annealing' in config and config['bootstrap_annealing']:
            config['checkpoint'] += '_%s' % config['bootstrap_annealing']
            if 'bootstrap_annealing_slope' in config and config['bootstrap_annealing_slope']:
                config['checkpoint'] += '_%g' % config['bootstrap_annealing_slope']
            else:
                config['bootstrap_annealing_slope'] = 1.0
        config['checkpoint'] += '_start_at=%i' % config['bootstrap_start_at']

    if 'lookahead' in config and config['lookahead']:
        config['checkpoint'] += '_lookahead_%i_%g' % (config['la_steps'], config['la_alpha'])

    # if 'dataset_tar' in config and config['dataset_tar']:
    #     config['checkpoint'] = config['dataset_tar'].split('.')[0] + '_' + config['checkpoint']
    if 'augment' in config and config['augment']:
        if config['augment'] in ['pgd', 'aa', 'gaussian']:
            if config['ad_aug_eps_train'] > 0 or config['ad_aug_eps_test'] > 0:
                config['checkpoint'] = 'ad_aug_%s_eps=%i_epst=%i_' % (config['augment'], config['ad_aug_eps_train'], config['ad_aug_eps_test']) + config['checkpoint']
            else:
                # If no augmentation in both train and test, the augment adversary doesn't matter
                config['checkpoint'] = 'ad_aug_eps=%i_epst=%i_' % (config['ad_aug_eps_train'], config['ad_aug_eps_test']) + config['checkpoint']

        if config['augment'] == 'mixup':
            checkpoint_head = 'mixup_aug_ratio=%g_ratiot=%g_' % (config['mixup_aug_ratio_train'], config['mixup_aug_ratio_test']) 
            if 'mixup_aug_target' in config and config['mixup_aug_target'] == 'true':
                checkpoint_head += 'true_'
            config['checkpoint'] = checkpoint_head + config['checkpoint']


    ## --------- adversary
    if config['adversary']:
        config['checkpoint'] += '_ad'
        config['checkpoint'] += '_%s' % config['adversary']
        if config['adversary'] in ['pgd', 'trades', 'mart', 'fat', 'gairat']:
            config['checkpoint'] += '_%i' % config['pgd_iter']
        if config['eps'] != 8:
            config['checkpoint'] += '_eps=%i' % config['eps']
        if 'pgd' in config['adversary'] or 'fgsm' in config['adversary']:
            # no para alpha if adversary == trades, llr, or mart
            if config['alpha'] != 0.5:
                config['checkpoint'] += ('_alpha=%g' % config['alpha']).replace('.', '_')
        if config['adversary'] == 'fat':
            if max(config['fat_taus']) != 2:
                config['checkpoint'] += '_mtau=%i' % max(config['fat_taus'])
        if 'target' in config and config['target']:
            config['checkpoint'] += '_target=%s' % config['target']
        if not config['rand_init']:
            config['checkpoint'] += '_zeroinit'
        if 'ad_soft_label' in config and config['ad_soft_label']:
            config['checkpoint'] += '_softlabel'

    if config['scheduler'] != 'multistep':
        config['checkpoint'] += '_%s' % config['scheduler']
        if 'cyclic' in config['scheduler']:
            config['checkpoint'] += '_%g_%g' % (config['lr'], config['lr_max'])
        if config['scheduler'] == 'cosine':
            config['checkpoint'] += '_lr=%g' % config['lr'] # max lr, also initial lr
        if config['scheduler'] == 'cosine_restart':
            config['checkpoint'] += '_cycle=%g' % config['epoch_cycle']
    else:
        if config['lr'] != 0.1:
            config['checkpoint'] += ('_lr=%.e' % config['lr']).replace('.', '_')

    if config['batch_size'] != 128:
        config['checkpoint'] += ('_bs=%i' % config['batch_size']).replace('.', '_')
    if config['wd'] > 0:
        config['checkpoint'] += ('_wd=%g' % config['wd']).replace('.', '_')
    if config['momentum'] > 0:
        config['checkpoint'] += ('_mom=%g' % config['momentum']).replace('.', '_')
    if config['lmr'] > 0:
        config['checkpoint'] += ('_lmr=%g' % config['lmr']).replace('.', '_')
    if config['ad_test']:
        config['checkpoint'] += '_%s' % config['ad_test']
        if 'pgd' in config['ad_test']:
            config['checkpoint'] += '_%i' % config['pgd_iter_test']
        if config['eps_test'] != 8:
            config['checkpoint'] += '_epst=%i' % config['eps_test']
        if 'ad_soft_label_test' in config and config['ad_soft_label_test']:
            config['checkpoint'] += '_softlabel'
    if config['test']:
        config['checkpoint'] = 'test_' + config['checkpoint']
    del config['test']
    if config['classes']:
        config['checkpoint'] += '_%s' % ('-'.join(config['classes']))
    if config['trainsize']:
        config['checkpoint'] += '_ntrain=%i' % config['trainsize']
    if config['testsize']:
        config['checkpoint'] += '_ntest=%i' % config['testsize']
    if 'trainnoisyratio' in config and config['trainnoisyratio']:
        config['checkpoint'] += '_trainnoise=%g' % config['trainnoisyratio']
    if 'testnoisyratio' in config and config['testnoisyratio']:
        config['checkpoint'] += '_testnoise=%g' % config['testnoisyratio']
    if 'train_subset_path' in config and config['train_subset_path']:
        config['checkpoint'] += '_sub=%s' % config['train_subset_path'].split('/')[-1].split('.')[0]
    if 'noise_subset_path' in config and config['noise_subset_path']:
        config['checkpoint'] += '_noisesub=%s' % config['noise_subset_path'].split('/')[-1].rstrip('.npy')
    if 'eval_subset_path' in config and config['eval_subset_path']:
        config['checkpoint'] += '_extra_eval'
    if 'alpha_sample_path' in config and config['alpha_sample_path']:
        # config['checkpoint'] += '_weighted'
        config['checkpoint'] += '_alpha=%s' % config['alpha_sample_path'].split('/')[-1].split('.')[0]
        # if 'label_smoothing' not in config and 'loss_flooding' not in config and config['adversary'] != 'trades':
        if 'pgd' in config['adversary'] or 'fgsm' in config['adversary']:
            if config['alpha'] != 1.:
                regex = r'(?<=weights_)(\d_\d+)(?=\.npy)'
                alpha_ = float(re.findall(regex, config['alpha_sample_path'])[0].replace('_', '.'))
                assert(alpha_ == config['alpha']), 'alpha sample path %s not consistent with alpha %g' % (config['alpha_sample_path'], config['alpha'])
    if 'lambda_sample_path' in config and config['lambda_sample_path']:
        config['checkpoint'] += '_lambda=%s' % config['lambda_sample_path'].split('/')[-1].split('.')[0]
    if 'weps_sample_path' in config and config['weps_sample_path']:
        config['checkpoint'] += '_weps=%s' % config['weps_sample_path'].split('/')[-1].split('.')[0]
    if 'num_iter_sample_path' in config and config['num_iter_sample_path']:
        config['checkpoint'] += '_witer=%s' % config['num_iter_sample_path'].split('/')[-1].split('.')[0]

    if 'label_smoothing' in config and config['label_smoothing']:
        if 'reg_sample_path' in config and config['reg_sample_path']:
            # use hyper parameters in weights
            # config['checkpoint'] += '_ls'
            config['checkpoint'] += '_ls=%s' % config['reg_sample_path'].split('/')[-1].split('.')[0]
        else:
            config['checkpoint'] += '_ls=%g' % config['label_smoothing']
    if 'loss_flooding' in config and config['loss_flooding']:
        if 'reg_sample_path' in config and config['reg_sample_path']:
            config['checkpoint'] += '_lf'
        else:
            config['checkpoint'] += '_lf=%g' % config['loss_flooding']

    if 'amp' in config and config['amp']:
        raise KeyError('AMP is disabled -- causing gradient masking!')
        config['checkpoint'] += '_amp'

    if 'manual_seed' in config and config['manual_seed']:
        config['checkpoint'] += '_seed=%i' % config['manual_seed']

    if 'model_seed' in config and config['model_seed']:
        config['checkpoint'] += '_modelSeed=%i' % config['model_seed']

    if config['suffix'] is not None:
        config['checkpoint'] += '_%s' % config['suffix']
    del config['suffix']

    if 'ext_model' in config and config['ext_model']:
        config['ext_model'] = os.path.join(os.getcwd(), 'checkpoints', config['ext_model'])
        config['checkpoint'] += '_extmodel'
    else:
        config['ext_model'] = ''

    path = os.path.join('checkpoints', config['checkpoint'])
    path = check_path(path, config)
    _, checkpoint = os.path.split(path)
    config['checkpoint'] = checkpoint
    # shutil.copy('models.py', path)
    # shutil.copy('config.yaml', path)
    shutil.copytree('src', os.path.join(path, 'src'))

    if config['resume']:
        config['resume_checkpoint'] = 'checkpoint.pth.tar'
        assert(os.path.isfile(os.path.join(path, config['resume_checkpoint']))), 'checkpoint %s not exists!' % config['resume_checkpoint']

    print("\n--------------------------- %s ----------------------------------" % config_file)
    for k, v in config.items():
        print('%s:'%k, v, type(v))
    print("---------------------------------------------------------------------\n")

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch-classification')
    parser.add_argument('--config', '-c', default='config.yaml', type=str, metavar='C', help='config file')
    args = parser.parse_args()

    config = read_config(args.config)
    with open('checkpoints/%s/para.json' % config['checkpoint'], 'w') as f:
        json.dump(config, f)

    # reveal the path to bash
    with open('tmp/path.tmp', 'w') as f:
        f.write(config['checkpoint'])


