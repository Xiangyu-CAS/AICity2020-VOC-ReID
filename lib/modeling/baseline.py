# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import numpy as np
import random
from torch import nn

from .backbones import build_backbone
from lib.layers.pooling import GeM
from lib.layers.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            #nn.init.constant_(m.weight, 1.0)
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def build_embedding_head(option, input_dim, output_dim, dropout_prob):
    reduce = None
    if option == 'fc':
        reduce = nn.Linear(input_dim, output_dim)
    elif option == 'dropout_fc':
        reduce = [nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                 ]
        reduce = nn.Sequential(*reduce)
    elif option == 'bn_dropout_fc':
        reduce = [nn.BatchNorm1d(input_dim),
                  nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                  ]
        reduce = nn.Sequential(*reduce)
    elif option == 'mlp':
        reduce = [nn.Linear(input_dim, output_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(output_dim, output_dim),
                 ]
        reduce = nn.Sequential(*reduce)
    else:
        print('unsupported embedding head options {}'.format(option))
    return reduce




class Baseline_reduce(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline_reduce, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.feature_dim = cfg.MODEL.EMBEDDING_DIM

        #self.reduce = nn.Linear(self.in_planes, self.feature_dim)
        self.reduce = build_embedding_head(cfg.MODEL.EMBEDDING_HEAD,
                                           self.in_planes, self.feature_dim,
                                           cfg.MODEL.DROPOUT_PROB)

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Arcface(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Cosface(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = AMSoftmax(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = CircleLoss(self.feature_dim, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)

        else:
            self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, return_featmap=False):
        featmap = self.base(x)
        if return_featmap:
            return featmap
        global_feat = self.gap(featmap)
        global_feat = global_feat.flatten(1)
        global_feat = self.reduce(global_feat)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat #global_feat  # global feature for triplet loss
        else:
            return feat


    def load_param(self, trained_path, skip_fc=True):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if skip_fc and 'classifier' in i:
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        #self.bottleneck = IBN(self.in_planes)

        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        #self.att = SpatialAttention2d(2048, 512)

    def forward(self, x, label=None, return_featmap=False):

        featmap = self.base(x)  # (b, 2048, 1, 1)
        #featmap = self.bottleneck(featmap)
        #featmap = self.att(featmap) * featmap
        if return_featmap:
            return featmap
        global_feat = self.gap(featmap)
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        global_feat = global_feat.flatten(1)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat#global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path, skip_fc=True):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if skip_fc and 'classifier' in i:
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])