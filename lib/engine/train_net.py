'''
@author: Xiangyu
'''
import os
import logging
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.reid_eval import evaluator

global ITER
ITER = 0

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def do_train(
        cfg,
        model,
        dataset,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE

    if device:
        #model.to(device)
        model.cuda()
        # Apex FP16 training
        if cfg.SOLVER.FP16:
            logging.getLogger("Using Mix Precision training")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    best_mAP = 0

    for epoch in range(start_epoch+1, cfg.SOLVER.MAX_EPOCHS+1):
        logger.info("Epoch[{}] lr={:.2e}"
                    .format(epoch, scheduler.get_lr()[0]))

        # freeze feature layer at warmup stage
        if cfg.SOLVER.FREEZE_BASE_EPOCHS != 0:
            if epoch < cfg.SOLVER.FREEZE_BASE_EPOCHS:
                logger.info("freeze base layers")
                frozen_feature_layers(model)
            elif epoch == cfg.SOLVER.FREEZE_BASE_EPOCHS:
                logger.info("open all layers")
                open_all_layers(model)

        train(model, dataset, train_loader, optimizer, loss_fn, epoch, cfg, logger)

        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            mAP = validate(model, dataset, val_loader, num_query, epoch, cfg, logger)
            if mAP >= best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))

        scheduler.step()
        torch.cuda.empty_cache()  # release cache
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                   os.path.join(output_dir, 'resume.pth.tar'))

    logger.info("best mAP: {:.1%}".format(best_mAP))
    torch.save(model.state_dict(), os.path.join(output_dir, 'final.pth'))
    os.remove(os.path.join(output_dir, 'resume.pth.tar'))


def train(model, dataset, train_loader, optimizer, loss_fn, epoch, cfg, logger):
    losses = AverageMeter()
    data_time = AverageMeter()
    model_time = AverageMeter()

    start = time.time()
    model.train()
    ITER = 0
    log_period = cfg.SOLVER.LOG_PERIOD
    data_start = time.time()
    for batch in train_loader:
        data_time.update(time.time() - data_start)
        input, target, _, _ = batch
        input = input.cuda()
        target = target.cuda()
        model_start = time.time()
        ITER += 1
        optimizer.zero_grad()
        score, feat = model(input, target)
        loss = loss_fn(score, feat, target)

        if cfg.SOLVER.FP16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_time.update(time.time() - model_start)
        losses.update(to_python_float(loss.data), input.size(0))

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, data time: {:.3f}s, model time: {:.3f}s"
                        .format(epoch, ITER, len(train_loader),
                                losses.val, data_time.val, model_time.val))
        data_start = time.time()
    end = time.time()
    logger.info("epoch takes {:.3f}s".format((end - start)))
    return


def validate(model, dataset, val_loader, num_query, epoch, cfg, logger):
    metric = evaluator(num_query, dataset, cfg, max_rank=50)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            data, pid, camid, img_path = batch
            data = data.cuda()
            feats = model(data)
            output = [feats, pid, camid, img_path]
            metric.update(output)
    cmc, mAP, _ = metric.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return mAP


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def frozen_feature_layers(model):
    for name, module in model.named_children():
        # if 'classifier' in name:
        #     module.train()
        #     for p in module.parameters():
        #         p.requires_grad = True
        # else:
        #     module.eval()
        #     for p in module.parameters():
        #         p.requires_grad = False
        if 'base' in name:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def open_all_layers(model):
    for name, module in model.named_children():
        module.train()
        for p in module.parameters():
            p.requires_grad = True