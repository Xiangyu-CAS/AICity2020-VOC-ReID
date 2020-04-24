# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths
