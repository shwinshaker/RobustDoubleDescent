#!./env python
# Weighted sample version

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ..preprocess import get_loaders
from ..utils import ParameterTracker, LipTracker, LrTracker, ResidualTracker, RobustTracker, ManifoldTracker, ExampleTracker, ConfTracker
from ..utils import save_checkpoint
from . import Tester, SWA
from ..adversary import AdTrainer
import time
import os

__all__ = ['train']


def train(config, net=None, loaders=None, criterion=None, optimizer=None, scheduler=None):

    time_start = time.time()

    swaer = None
    if hasattr(config, 'swa') and config.swa:
        swaer = SWA(loaders, net, optimizer, config, time_start)

    tester = Tester(loaders, net, optimizer, config, time_start, scheduler=scheduler)
    adtrainer = AdTrainer(loaders, net, optimizer=optimizer, criterion=criterion, config=config, time_start=time_start, swaer=swaer)

    # tracker switchs
    if config.lipTrack:
        lipLog = LipTracker(net, device=config.device)
    if config.paraTrack:
        paraLog = ParameterTracker(net)
    if config.lrTrack and isinstance(optimizer, optim.Adagrad):
        lrLog = LrTracker(net)
    if config.resTrack:
        resLog = ResidualTracker(net, device=config.device)
    if config.rbTrack:
        rbLog = RobustTracker(net, loaders, config, time_start)
    if config.mrTrack:
        mrLog = ManifoldTracker(net, criterion, config=config)
    if hasattr(config, 'confTrack') and config.confTrack:
        confLog = ConfTracker(net, loaders, time_start, config)

    if config.lmr > 0:
        raise NotImplementedError('Deprecated')

    for epoch in range(config.epoch_start, config.epochs):        
        net.train()
        for i, (inputs, labels, weights) in enumerate(loaders.trainloader, 0):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            loss = adtrainer._loss(inputs, labels, weights, epoch=epoch)

            # update net
            # net.zero_grad() # should be identical
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test mode
        net.eval() 
        tester.update(epoch, i)
        if hasattr(config, 'swa') and config.swa:
            swaer.update(epoch)

        adtrainer.update(epoch, i)
        adtrainer.reset(epoch)

        # trackers
        if config.paraTrack:
            paraLog.update(net, epoch, i)
        if config.lipTrack:
            lipLog.update(epoch, i)
        if config.lrTrack and isinstance(optimizer, optim.Adagrad):
            lrLog.update(epoch, i, optimizer)
        if config.resTrack:
            resLog.update(epoch, i)
        if config.rbTrack:
            rbLog.update(epoch)
        if config.mrTrack:
            mrLog.record(epoch, i, config.lmr)
        if hasattr(config, 'confTrack') and config.confTrack:
            confLog.update(epoch)

        if scheduler:
            scheduler.step()

        save_checkpoint(epoch, net, optimizer, scheduler)

        if config.save_interval:
            # save model sequentially
            if epoch % config.save_interval == 0:
                torch.save(net.state_dict(), 'model-%s.pt' % epoch)

    torch.save(net.state_dict(), 'model.pt')

    tester.close()
    adtrainer.close()
    if hasattr(config, 'swa') and config.swa:
        swaer.close()
    if config.paraTrack:
        paraLog.close()
    if config.lipTrack:
        lipLog.close()
    if config.lrTrack and isinstance(optimizer, optim.Adagrad):
        lrLog.close()
    if config.resTrack:
        resLog.close()
    if config.rbTrack:
        rbLog.close()
    if config.mrTrack:
        mrLog.close()
    if hasattr(config, 'confTrack') and config.confTrack:
        confLog.close()

