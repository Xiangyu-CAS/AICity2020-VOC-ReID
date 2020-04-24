# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
import time
import torch
import torch.nn as nn
from lib.utils.reid_eval import evaluator

def inference(
        cfg,
        model,
        val_loader,
        num_query,
        dataset
):
    device = cfg.MODEL.DEVICE
    model.to(device)
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    metric = evaluator(num_query, dataset, cfg, max_rank=50)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in val_loader:
            data, pid, camid, img_path = batch
            data = data.cuda()
            feats = model(data)
            if cfg.TEST.FLIP_TEST:
                data_flip = data.flip(dims=[3])  # NCHW
                feats_flip = model(data_flip)
                feats = (feats + feats_flip) / 2
            output = [feats, pid, camid, img_path]
            metric.update(output)
    end = time.time()
    logger.info("inference takes {:.3f}s".format((end - start)))
    torch.cuda.empty_cache()
    cmc, mAP, indices_np = metric.compute()
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return indices_np

def select_topk(indices, query, gallery, topk=10):
    results = []
    for i in range(indices.shape[0]):
        ids = indices[i][:topk]
        results.append([query[i][0]] + [gallery[id][0] for id in ids])
    return results


def extract_features(cfg, model, loader):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data, pid, camid, img_path = batch
            data = data.cuda()
            feat = model(data)
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats