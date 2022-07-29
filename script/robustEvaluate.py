#!./env python

import torch
import numpy as np
import os
import argparse

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn

from src.preprocess import get_loaders
from src.analyses import get_net
from src.utils import str2bool
from src.utils import accuracy, Dict2Obj, AverageMeter

from RayS.general_torch_model import GeneralTorchModel
from RayS.RayS import RayS

class RobustEvaluator:

    available_metrics = ['CW', 'PGD', 'PGD-100', 'PGD-1000', 'FGSM', 'DeepFool',
                         'SquareAttack', 'BoundaryAttack', 'HopSkipJump', 'RayS']

    def __init__(self, net, loaders, metrics, save_path, config, src_net=None):
        assert(all([m in self.available_metrics for m in metrics]))
        self.config = config
        self.device = config.device

        self.net = net
        if src_net is not None:
            print('> Transfer attack enabled.')
            # transfer attack from a surrogate model
            self.src_net = src_net
        else:
            # attack model itself
            self.src_net = net
        self.loaders = loaders
        self.loader = loaders.testloader
        self.metrics = metrics
        self.save_path = save_path

        self.__art_setup()

    def eval_robust_acc(self):
        meters = dict([(metric, AverageMeter()) for metric in self.metrics])
        for e, (inputs, labels, weights) in enumerate(self.loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.config.virtual:
                # use model's prediction
                labels = self.net(inputs).max(1)[1].detach()
            inputs = self.__to_art(inputs)
            labels = self.__to_art_labels(labels)
            for metric in self.metrics:
                inputs_ad = self.methods[metric](inputs, labels)
                with torch.no_grad():
                    outputs = self.net(inputs_ad)
                    meters[metric].update(accuracy(outputs, self.__from_art_labels(labels))[0].item(), inputs.size(0))
                # save correct ids
                correct_ids = self.get_correct_ids(outputs, self.__from_art_labels(labels), weights['index'].to(self.device))
                self.__append_correct_ids(correct_ids, metric=metric)
            print_string = ' -- '.join(['%s: %.4f' % (metric, meters[metric].avg) for metric in metrics])
            print('----------- [%i/%i] --- %s ------' % (e, len(self.loader), print_string))

        robust_results = dict([(metric, meters[metric].avg) for metric in self.metrics])
        torch.save(robust_results, self.save_path)
        return robust_results

    def __to_art(self, tensor):
        # from torch type (cuda tensor) to art type (cpu tensor)
        return self.denormalize(tensor).cpu().detach()

    def __to_art_labels(self, tensor):
        return tensor.cpu().numpy()

    def __from_art(self, array):
        # from art type (cpu tensor) to torch type (cuda tensor)
        return self.normalize(torch.tensor(array).cuda())

    def __from_art_labels(self, array):
        return torch.tensor(array).cuda()

    def get_correct_ids(self, outputs, labels, indices):
        _, preds = outputs.max(1)
        return indices[preds.squeeze().eq(labels)].cpu().numpy()

    def __append_correct_ids(self, array, metric):
        file_name = os.path.splitext(self.save_path)[0] + '_%s_correct_ids.npy' % metric
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                record = np.load(f, allow_pickle=True)
            record = np.append(record, array)
        else:
            record = np.array(array)
        with open(file_name, 'wb') as f:
            np.save(f, record, allow_pickle=True)

    def __art_setup(self):

        # ART interface to pytorch
        from src.preprocess import dataset_stats
        mean = np.array(dataset_stats[self.config.dataset]['mean']).astype(np.float32)
        std = np.array(dataset_stats[self.config.dataset]['std']).astype(np.float32)

        from src.utils import DeNormalizer, Normalizer
        self.denormalize = DeNormalizer(mean, std, self.loaders.n_channel, self.config.device)
        self.normalize = Normalizer(mean, std, self.loaders.n_channel, self.config.device)

        # create an ART instance wrapper
        from art.estimators.classification import PyTorchClassifier
        ctr = nn.CrossEntropyLoss()
        self.classifier = PyTorchClassifier(model=self.src_net,
                                            loss=ctr,
                                            input_shape=tuple(self.loaders.shape),
                                            nb_classes=self.loaders.num_classes,
                                            channels_first=True,
                                            preprocessing=(mean.reshape(self.loaders.n_channel, 1, 1),
                                                           std.reshape(self.loaders.n_channel, 1, 1)),
                                            clip_values=(0, 1),
                                            device_type='gpu')


        # create methods
        self.methods = dict()
        if 'RayS' in self.metrics:
            epsilon = self.config.eps/255
            max_iter = 10000
            from RayS.general_torch_model import GeneralTorchModel
            from RayS.RayS import RayS
            torch_model = GeneralTorchModel(self.src_net, n_class=10, im_mean=mean, im_std=std)
            attack = RayS(torch_model, epsilon=epsilon)
            def __get_RayS(inputs, labels):
                inputs, labels = inputs.cuda(), torch.tensor(labels).cuda()
                inputs_ad, queries, adbd, succ = attack(inputs, labels, query_limit=max_iter)
                delta = torch.clamp(inputs_ad - inputs, -epsilon, epsilon)
                inputs_ad = inputs + delta
                inputs_ad = self.normalize(inputs_ad)
                return inputs_ad
            self.methods['RayS'] = __get_RayS


        if 'FGSM' in self.metrics:
            from art.attacks.evasion import FastGradientMethod
            self.FGSM = FastGradientMethod(self.classifier,
                                           norm=np.inf,
                                           targeted=False,
                                           eps_step=1/255,
                                           minimal=True)
            def __get_FGSM(inputs):
                return self.__from_art(self.FGSM.generate(inputs))
            self.methods['FGSM'] = __get_FGSM

        if 'PGD' in self.metrics:
            from art.attacks.evasion import ProjectedGradientDescent
            self.PGD = ProjectedGradientDescent(self.classifier,
                                                norm=np.inf,
                                                eps=self.config.eps/255,
                                                eps_step=2/255,
                                                max_iter=10,
                                                targeted=False,
                                                num_random_init=0,
                                                batch_size=self.config.batch_size,
                                                random_eps=False,
                                                verbose=True)
            def __get_PGD(inputs, labels):
                inputs_ad = self.PGD.generate(inputs, labels)
                return self.__from_art(inputs_ad)
            self.methods['PGD'] = __get_PGD

        if 'PGD-1000' in self.metrics:
            from art.attacks.evasion import ProjectedGradientDescent
            self.PGD_ = ProjectedGradientDescent(self.classifier,
                                                 norm=np.inf,
                                                 eps=self.config.eps/255,
                                                 eps_step=2/255,
                                                 max_iter=1000,
                                                 targeted=False,
                                                 num_random_init=0,
                                                 batch_size=self.config.batch_size,
                                                 random_eps=False,
                                                 verbose=True)
            def __get_PGD_(inputs, labels):
                inputs_ad = self.PGD_.generate(inputs, labels)
                return self.__from_art(inputs_ad)
            self.methods['PGD-1000'] = __get_PGD_

        if 'DeepFool' in self.metrics:
            raise NotImplementedError()

        if 'CW' in self.metrics:
            from art.attacks.evasion import CarliniLInfMethod
            # self.CW = CarliniLInfMethod(self.classifier, targeted=False, confidence=0)
            self.CW = CarliniLInfMethod(self.classifier,
                                        confidence=0,
                                        targeted=False,
                                        learning_rate=0.01,
                                        max_iter=10,
                                        max_halving=5,
                                        max_doubling=5,
                                        eps=self.config.eps/255,
                                        batch_size=self.config.batch_size,
                                        verbose=True)
            def __get_CW(inputs, labels):
                return self.__from_art(self.CW.generate(inputs, labels, verbose=False))
            self.methods['CW'] = __get_CW

        if 'SquareAttack' in self.metrics:
            from art.attacks.evasion import SquareAttack
            self.SquareAttack = SquareAttack(self.classifier,
                                             norm=np.inf,
                                             max_iter=5000,
                                             eps=self.config.eps/255,
                                             p_init=0.8,
                                             nb_restarts=1,
                                             batch_size=self.config.batch_size,
                                             verbose=True)
            def __get_Square(inputs, labels):
                return self.__from_art(self.SquareAttack.generate(inputs, labels))
            self.methods['SquareAttack'] = __get_Square

        if 'BoundaryAttack' in self.metrics:
            from art.attacks.evasion import BoundaryAttack
            self.BoundaryAttack = BoundaryAttack(self.classifier,
                                                 # batch_size=self.config.batch_size,
                                                 targeted=False,
                                                 delta=0.01,
                                                 epsilon=0.01,
                                                 step_adapt=0.667,
                                                 max_iter=10, # 5000,
                                                 num_trial=25,
                                                 sample_size=20,
                                                 init_size=100,
                                                 min_epsilon= None,
                                                 verbose=True)

            def __get_Boundary(inputs, labels):
                return self.__from_art(self.BoundaryAttack.generate(inputs, labels))
            self.methods['BoundaryAttack'] = __get_Boundary

        if 'HopSkipJump' in self.metrics:
            from art.attacks.evasion import HopSkipJump
            self.HopSkipJump = HopSkipJump(self.classifier,
                                           targeted=False,
                                           norm=np.inf,
                                           max_iter=10, # 50,
                                           max_eval=25, # 10000,
                                           init_eval=25, # 100,
                                           init_size=100,
                                           verbose=True)

            def __get_HopSkipJump(inputs, labels):
                return self.__from_art(self.HopSkipJump.generate(inputs, labels))
            self.methods['HopSkipJump'] = __get_HopSkipJump


def robust_evaluate(model, depth, width, state='last',
                    virtual=False, eps=8,
                    path='.', save_path=None, gpu_id='0',
                    metrics=['PGD', 'CW', 'SquareAttack'], transfer=False, transfer_model='wrn-28-10',
                    dataset='cifar10', data_dir='/home/chengyu/Initialization/data'):

    print('>>>>>>>>>>> set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('>>>>>>>>>>> get loader..')
    config = dict()
    config['dataset'] = dataset
    config['data_dir'] = data_dir
    config['batch_size'] = 256 # 128
    config['traintest'] = False # True
    config['shuffle_train_loader'] = False # if break, maintain loader order when continue
    config['random_augment'] = False # Produce an adversarial counterpart of the original image
    config['device'] = device
    config['virtual'] = virtual
    config['eps'] = eps
    config = Dict2Obj(config)

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

    if save_path is None:
        save_path = 'log'
        if transfer:
            if transfer_model == 'wrn-28-10':
                save_path += '_transfer'
            elif transfer_model == 'wrn-28-5':
                save_path += '_transfer1'
            elif transfer_model == 'pre18':
                save_path += '_transfer2'
            elif transfer_model == 'mart':
                save_path += '_transfer_mart'
            elif transfer_model == 'trades':
                save_path += '_transfer_trades'
            else:
                raise ValueError('transfer model not defined %s' % transfer_model)
        save_path += '_robustness_%s' % state
        save_path += '_' + '-'.join(metrics)
        save_path += '_eps=%g' % eps
        if virtual:
            save_path += '_virtual'
        save_path += '.pt'
        save_path = os.path.join(path, save_path)

    net = get_net(path,
                  num_classes=loaders.num_classes,
                  n_channel=loaders.n_channel,
                  feature=None,
                  model=model,
                  depth=depth,
                  width=width,
                  state=model_state,
                  device=device)

    src_net = None
    if transfer:
        if transfer_model == 'wrn-28-10':
            src_path = 'checkpoints/sgd_wrn-28-10_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
            model = 'wrn'
            width = 10
        elif transfer_model == 'wrn-28-5':
            src_path = 'checkpoints/sgd_wrn-28-5_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
            model = 'wrn'
            width = 5
        elif transfer_model == 'pre18':
            src_path = 'checkpoints/sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10-0'
            model = 'PreActResNet18'
        elif transfer_model == 'mart':
            src_path = 'checkpoints/sgd_PreActResNet18_gain=1_0_ad_mart_10_wd=0_0005_mom=0_9_pgd_10-1'
            model = 'PreActResNet18'
        elif transfer_model == 'trades':
            src_path = 'checkpoints/sgd_PreActResNet18_gain=1_0_ad_trades_10_wd=0_0005_mom=0_9_pgd_10'
            model = 'PreActResNet18'
        src_net = get_net(src_path,
                          num_classes=loaders.num_classes,
                          n_channel=loaders.n_channel,
                          feature=None,
                          model=model,
                          depth=28,
                          width=width,
                          state='best_model.pt',
                          device=device)

    print('>>>>>>>>>>> start evaluating..')
    evaluator = RobustEvaluator(net, loaders, metrics, save_path, config, src_net=src_net)
    evaluator.eval_robust_acc()

    print('>>>>>>>>>>> Done.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cifar10', type=str, help='dataset')
    parser.add_argument("--eps", default=8, type=float, help='perturbation radius')
    parser.add_argument('-m', "--model", default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=20, type=int, help='model depth')
    parser.add_argument('--width', default=64, type=int, help='model width')
    parser.add_argument("--virt", type=str2bool, nargs='?', const=True, default=False, help="use model's own prediction as label?")
    parser.add_argument("-t", "--transfer", type=str2bool, nargs='?', const=True, default=False, help="transfer attack or not?")
    # parser.add_argument("--metrics", default=['PGD', 'CW', 'SquareAttack'], type=list, help="eval metrics")
    parser.add_argument('-d', "--state", default='last', type=str, help='model state')
    parser.add_argument("-p", "--path", type=str, help="model path")
    parser.add_argument("-sp", "--save_path", type=str, help="save path")
    parser.add_argument("-g", "--gpu", default='0', type=str, help="gpu_id")
    args = parser.parse_args()

    metrics = ['PGD', 'PGD-1000', 'CW']
    transfer_model = 'wrn-28-10' # None

    robust_evaluate(model=args.model, depth=args.depth, width=args.width, state=args.state,
                    metrics=metrics, virtual=args.virt, eps=args.eps,
                    dataset=args.dataset,
                    transfer=args.transfer, transfer_model=transfer_model,
                    path=args.path, save_path=args.save_path, gpu_id=args.gpu)


