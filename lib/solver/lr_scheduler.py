# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch
import math


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=10,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


'''
Bag of Tricks for Image Classification with Convolutional Neural Networks
'''
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs=10,
        eta_min=1e-7,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs - 1
        self.eta_min=eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        else:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
        return lr


class CosineStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        step_epochs=2,
        gamma=0.3,
        eta_min=0,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs
        self.eta_min=eta_min
        self.step_epochs = step_epochs
        self.gamma = gamma
        self.last_cosine_lr = 0
        super(CosineStepLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.max_epochs - self.step_epochs:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch) / (self.max_epochs - self.step_epochs))) / 2
                    for base_lr in self.base_lrs]
            self.last_cosine_lr = lr
        else:
            lr = [self.gamma ** (self.step_epochs - self.max_epochs + self.last_epoch + 1) * base_lr for base_lr in self.last_cosine_lr]

        return lr


class CyclicCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 cycle_epoch,
                 cycle_decay=0.7,
                 last_epoch=-1):
        self.cycle_decay = cycle_decay
        self.cycle_epoch = cycle_epoch
        self.cur_count = 0
        super(CyclicCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.cur_count = (self.last_epoch + 1) // self.cycle_epoch
        decay = self.cycle_decay ** self.cur_count
        return [base_lr * decay *
         (1 + math.cos(math.pi * (self.last_epoch % self.cycle_epoch) / self.cycle_epoch)) / 2
         for base_lr in self.base_lrs]



def build_lr_scheduler(optimizer, lr_scheduler, cfg, last_epoch):
    if lr_scheduler == 'warmup_multi_step':
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, last_epoch=last_epoch)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(cfg.SOLVER.MAX_EPOCHS), last_epoch=last_epoch)
    elif lr_scheduler == 'warmup_cosine':
        scheduler = WarmupCosineLR(optimizer, max_epochs=float(cfg.SOLVER.MAX_EPOCHS),
                                   warmup_epochs=cfg.SOLVER.WARMUP_ITERS, last_epoch=last_epoch)
    elif lr_scheduler == 'cyclic_cosine':
        scheduler = CyclicCosineLR(optimizer, cfg.SOLVER.CYCLE_EPOCH)
    elif lr_scheduler == 'cosine_step':
        scheduler = CosineStepLR(optimizer, max_epochs=float(cfg.SOLVER.MAX_EPOCHS), last_epoch=last_epoch)
    else:# multi-steps as default
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA, last_epoch=last_epoch)

    return scheduler
