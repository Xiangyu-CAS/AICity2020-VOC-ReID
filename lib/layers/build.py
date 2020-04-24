# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .metric_learning import ContrastiveLoss

def make_loss(cfg, num_classes):    # modified by gu
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_func = TripletLoss(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'contrastive':
        metric_loss_func = ContrastiveLoss(cfg.SOLVER.MARGIN)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'none':
        def metric_loss_func(feat, target):
            return 0
    else:
        print('got unsupported metric loss type {}'.format(
            cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    else:
        id_loss_func = F.cross_entropy

    def loss_func(score, feat, target):
        return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
               cfg.MODEL.TRIPLET_LOSS_WEIGHT * metric_loss_func(feat, target)
    return loss_func