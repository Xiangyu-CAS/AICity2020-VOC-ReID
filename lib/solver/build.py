# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn as nn

from .ranger import Ranger
from .swa import SWA


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if 'classifier' in key: # different learning rate for initialized fc layers
            lr = cfg.SOLVER.FC_LR_FACTOR * lr
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Ranger':
        optimizer = Ranger(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SWA':
        print('using SWA')
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
        optimizer = SWA(optimizer, swa_start=0, swa_freq=1)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer
