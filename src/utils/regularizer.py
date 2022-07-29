#!./env python

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LabelSmoothingCrossEntropy', 'LossFloodingCrossEntropy']

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.reduction = reduction
        self.smoothing_ = smoothing

    def forward(self, x, target, weights=None): # named 'weights' to allow overloading
        smoothing = weights
        if smoothing is None:
            smoothing = self.smoothing_
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise KeyError('Not supported reduction: %s' % self.reduction)

class LossFloodingCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', flooding=0.1):
        super(LossFloodingCrossEntropy, self).__init__()
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction='none') # take care of reduction in forwarding
        self.flooding_ = flooding

    def forward(self, x, target, weights=None):
        flooding = weights
        if flooding is None:
            flooding = self.flooding_
        loss = self.criterion(x, target)
        loss = (loss - flooding).abs() + flooding
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise KeyError('Not supported reduction: %s' % self.reduction)
