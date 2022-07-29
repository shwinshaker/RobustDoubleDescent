#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import re

# from torch.cuda.amp import autocast

__all__ = ['test', 'AverageMeter', 'GroupMeter', 'accuracy', 'alignment', 'criterion_r', 'ce_soft', 'mse_one_hot', 'Confidence']


def test(testloader, net, criterion, config, classes=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    net.eval()

    if hasattr(config, 'class_eval') and config.class_eval:
        top1_class = GroupMeter(classes)
    
    for i, tup in enumerate(testloader, 0):
        if len(tup) == 2:
            inputs, labels = tup
        else:
            inputs, labels, _ = tup
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        # if hasattr(config, 'amp') and config.amp:
        #     with autocast():
        #         outputs = net(inputs)
        #         loss = criterion(outputs, labels)
        # else:
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, labels)

        prec1, = accuracy(outputs.data, labels.data)

        losses.update(loss.item(), inputs.size(0))        
        top1.update(prec1.item(), inputs.size(0))

        if hasattr(config, 'class_eval') and config.class_eval:
            top1_class.update(outputs, labels)

    extra_metrics = dict()
    if hasattr(config, 'class_eval') and config.class_eval:
        extra_metrics['class_acc'] = top1_class.output_group()
        
    return losses.avg, top1.avg, extra_metrics


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


class GroupMeter:
    """
        measure the accuracy of each class
    """
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.meters = [AverageMeter() for _ in range(self.num_classes)]

    def update(self, out, las):
        _, preds = out.topk(1, 1, True, True)
        preds = preds.squeeze()
        for c in range(self.num_classes):
            num_c = (las == c).sum().item()
            if num_c == 0:
                continue
            acc = ((preds == las) & (las == c)).sum() * 100. / num_c
            self.meters[c].update(acc.item(), num_c)

    def output(self):
        return np.mean(self.output_group())

    def output_group(self):
        return [self.meters[i].avg for i in range(self.num_classes)]

    def pprint(self):
        print('')
        for i in range(self.num_classes):
            print('%10s: %.4f' % (self.classes[i], self.meters[i].avg))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    if len(target.size()) == 1:
        pass
    elif len(target.size()) == 2:
        # soft label
        _, target = target.max(1)
    else:
        raise TypeError(target.size())

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def alignment(output1, output2):
    batch_size = output1.size(0)

    _, pred1 = output1.topk(1, dim=1, largest=True)
    pred1 = pred1.t()
    _, pred2 = output2.topk(1, dim=1, largest=True)
    pred2 = pred2.t()
    correct = pred1.eq(pred2)
    return correct.view(-1).float().sum(0).mul_(100.0 / batch_size)


def ce_soft(reduction='mean', num_classes=10, soft_label=True, temperature=1.0):
    def ce(outputs, labels):
        assert(outputs.size(1) == num_classes), (outputs.size(), num_classes)
        if not soft_label:
            if len(labels.size()) > 1:
                _, labels = torch.max(labels, 1)
            return torch.nn.functional.cross_entropy(outputs, labels, reduction=reduction)
        
        assert(len(labels.size()) == 2), ('soft labels required, got size: ', labels.size())
        log_probas = torch.nn.functional.log_softmax(outputs / temperature, dim=1)
        nll = -(log_probas * labels) * temperature**2 # normalize
        loss = nll.sum(dim=1)
        if reduction == 'mean':
            return loss.mean()
        return loss
    return ce


def mse_one_hot(reduction='mean', num_classes=10, soft_label=False):
    def mse(outputs, labels):
        assert(outputs.size(1) == num_classes), (outputs.size(), num_classes)
        probas = torch.nn.functional.softmax(outputs, dim=1)
        if not soft_label:
            if len(labels.size()) == 1:
                # provided labels are hard label
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            else:
                # provided labels are soft label
                _, labels = torch.max(labels, 1)
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        if reduction == 'mean':
            return nn.MSELoss(reduction=reduction)(probas, labels) * num_classes # mse_loss in pytorch average over dimension too
        return nn.MSELoss(reduction='none')(probas, labels).sum(dim=1) # reduce class dimension, keep batch dimension
    return mse

def criterion_r(output1, output2, c=None):

    if isinstance(c, nn.CrossEntropyLoss):
        # https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/17
        def cross_entropy(pred, soft_targets):
            logsoftmax = nn.functional.log_softmax
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred, dim=1), 1))
        return cross_entropy(output1, output2)
        
    raise NotImplementedError(type(c))


class Confidence:

    __available_metrics = ['NLL', 'Brier', 'ECE', 'AURC', 'TV', 'KL'] # TV and KL require true label distribution
    __instance_metrics = ['NLL', 'Brier', 'TV', 'KL']
    __soft_metrics = ['TV', 'KL']
    __macro_metrics = ['ECE', 'AURC']

    def __init__(self, metrics, num_classes, device, instancewise=False):
        assert(all([m.split('-')[0] in self.__available_metrics for m in metrics]))
        self.metrics = metrics
        self.num_classes = num_classes
        self.device = device

        self.instancewise = instancewise

    def evaluate(self, net, loader, num_ex=None):
        meters = dict([(metric, AverageMeter()) for metric in self.metrics if metric in self.__instance_metrics])
        if self.instancewise:
            raw_results = dict()
            for metric in self.metrics:
                if metric in self.__instance_metrics:
                    raw_results[metric] = np.zeros(num_ex).astype(np.float16)
        list_corrects = []
        list_softmaxs = []
        for e, (inputs, labels, weights) in enumerate(loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = net(inputs)
                for metric in self.metrics:
                    if metric in self.__instance_metrics:
                        if metric in self.__soft_metrics:
                            labels_ = weights['label_distribution'].to(self.device)
                        else:
                            labels_ = labels
                        if self.instancewise:
                            confs = getattr(self, metric.lower())(outputs, labels_, reduction=False)
                            raw_results[metric][weights['index']] = confs.cpu().numpy().astype(np.float16)
                            conf_ = confs.mean().item()
                        else:
                            conf_ = getattr(self, metric.lower())(outputs, labels_, reduction=True)
                        meters[metric].update(conf_, inputs.size(0))
                # if 'NLL' in self.metrics:
                #     meters['NLL'].update(self.nll(outputs, labels).item(), inputs.size(0))
                # if 'Brier' in self.metrics:
                #     meters['Brier'].update(self.brier(outputs, labels).item(), inputs.size(0))
                # if 'TV' in self.metrics:
                #     soft_labels = weights['label_distribution'].to(self.device)
                #     meters['TV'].update(self.tv(outputs, soft_labels).item(), inputs.size(0))
                # if 'KL' in self.metrics:
                #     soft_labels = weights['label_distribution'].to(self.device)
                #     meters['KL'].update(self.kl(outputs, soft_labels).item(), inputs.size(0))
                _, preds = outputs.max(1)
                corrects = preds.eq(labels.view_as(preds))
                softmaxs = F.softmax(outputs, dim=1)
                list_corrects.append(corrects)
                list_softmaxs.append(softmaxs)
        corrects = torch.cat(list_corrects)
        softmaxs = torch.cat(list_softmaxs)

        results = dict([(metric, meters[metric].avg) for metric in meters])
        for metric in self.metrics:
            if metric == 'ECE':
                results[metric] = self.ece(softmaxs, corrects, bins=15)
            elif re.match('ECE-\d*', metric):
                n_bins = int(metric.split('-')[-1])
                results[metric] = self.ece(softmaxs, corrects, bins=n_bins)
        if 'AURC' in self.metrics:
            results['AURC'] = self.aurc(softmaxs, corrects)

        if self.instancewise:
            return results, raw_results
        return results

    def nll(self, outputs, targets, reduction=True):
        if not reduction:
            return F.cross_entropy(outputs, targets, reduction='none')
        return F.cross_entropy(outputs, targets).item()

    def brier(self, outputs, targets, reduction=True):
        if not reduction:
            mse = mse_one_hot(reduction='none', num_classes=self.num_classes, soft_label=False)
            return mse(outputs, targets)
        mse = mse_one_hot(num_classes=self.num_classes, soft_label=False)
        return mse(outputs, targets).item()

    def tv(self, outputs, soft_targets, reduction=True):
        if not reduction:
            return F.l1_loss(F.softmax(outputs, dim=1), soft_targets, reduction='none').sum(dim=1) / 2.0
        return F.l1_loss(F.softmax(outputs, dim=1), soft_targets, reduction='mean').item() * self.num_classes / 2.0
    
    def kl(self, outputs, soft_targets, reduction=True):
        if not reduction:
            return F.kl_div(F.log_softmax(outputs, dim=1), soft_targets, reduction='none').sum(dim=1)
        return F.kl_div(F.log_softmax(outputs, dim=1), soft_targets, reduction='batchmean').item()

    def ece(self, softmaxs, corrects, bins=15):
        bin_boundaries = torch.linspace(0, 1, bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        prob_maxs, _ = softmaxs.max(1)

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = prob_maxs.gt(bin_lower.item()) * prob_maxs.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0.0:
                accuracy_in_bin = corrects[in_bin].float().mean()
                avg_confidence_in_bin = prob_maxs[in_bin].mean()

                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        # print("ECE {0:.2f} ".format(ece.item()*100))
        return ece.item()

    def aurc(self, softmaxs, corrects):
        """
            AURC only cares about the ordinal ranking of the predictions,
                namely correctly predicted examples should always have higher confidence than incorrectly predicted examples
                see: `BIAS-REDUCED UNCERTAINTY ESTIMATION FOR DEEP NEURAL CLASSIFIERS`
        """
        prob_maxs, _ = softmaxs.max(1)

        # sort examples by confidence
        sort_prob_maxs, sort_indices = torch.sort(prob_maxs, descending=True)
        sort_corrects = corrects[sort_indices]
        # sort_values = sorted(zip(prob_maxs, corrects), key=lambda x: x[0], reverse=True)
        # sort_prob_maxs, sort_corrects = zip(*sort_values)

        # get risk-coverage curve
        risk_list = []
        risk = 0
        for i in range(len(sort_prob_maxs)):
            if sort_corrects[i] == 0:
                risk += 1
            risk_list.append(risk / (i + 1))

        aurc = torch.mean(torch.tensor(risk_list))
        # print("AURC {0:.2f}".format(aurc*1000))

        # r = risk_list[-1]
        # optimal_risk_area = r + (1 - r) * np.log(1 - r)
        # eaurc = risk_coverage_curve_area - optimal_risk_area
        # print("EAURC {0:.2f}".format(eaurc*1000))

        return aurc.item()