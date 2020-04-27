# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import json
import os.path as osp
import os
import time
from .post_process import *

def eval_func(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = indices.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    total_camids = len(set(g_camids))
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        if total_camids == 1:# for NAIC
            remove = (g_pids[order] == q_pid) & (g_camids[order] != q_camid)
        else: # for others like market1501
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.array(tmp_cmc) / (np.arange(len(tmp_cmc)) + 1.)
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class evaluator(object):
    def __init__(self, num_query, dataset, cfg, max_rank=50):
        super(evaluator, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = cfg.TEST.FEAT_NORM
        self.query_expansion = cfg.TEST.QUERY_EXPANSION
        self.query_expansion_topk = 6
        self.do_DBA = cfg.TEST.DO_DBA
        self.dataset = dataset
        self.do_rerank = cfg.TEST.DO_RERANK
        self.rerank_param = cfg.TEST.RERANK_PARAM
        self.cfg = cfg

        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, output):
        feat, pid, camid, img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        if self.do_DBA:
            feats = database_aug(feats, top_k=6)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_names = [os.path.basename(img_path) for img_path in self.img_paths[self.num_query:]]

        if self.query_expansion:
            qf = average_query_expansion(qf, feats, top_k=6)

        if self.cfg.TEST.TRACK_AUG:
            gf = track_aug(gf, self.dataset.test_tracks, self.img_paths[self.num_query:])

        #qf, gf = pca_whiten(qf, gf)
        if self.cfg.TEST.USE_VOC:
            print('using VOC-ReID')
            cam_dist = np.load(self.cfg.TEST.CAM_DIST_PATH)
            ori_dist = np.load(self.cfg.TEST.ORI_DIST_PATH)
        else:
            cam_dist = None
            ori_dist = None

        if self.do_rerank:
            distmat_np = re_ranking(qf, gf,
                                    k1=self.rerank_param[0],
                                    k2=self.rerank_param[1],
                                    lambda_value=self.rerank_param[2], USE_VOC=self.cfg.TEST.USE_VOC, cam_dist=cam_dist, ori_dist=ori_dist)
        else:
            distmat, indices = comput_distmat(qf, gf)
            distmat_np = distmat.cpu().numpy()

        # track_idxs = generate_track_idxs(g_names, self.dataset.test_tracks)
        # distmat_track_np = generate_track_distmat(distmat_np, track_idxs)
        # np.save(os.path.dirname(self.cfg.TEST.WEIGHT) + '/distmat_track',
        #         distmat_track_np)

        # cam_distmat = np.load('./output/aicity20/experiments/ReCamID/distmat.npy')
        # ori_distmat = np.load('./output/aicity20/experiments/ReOriID/distmat.npy')
        #cam_distmat = np.load('./output/aicity20/0410-test/ReCamID/distmat.npy')
        #ori_distmat = np.load('./output/aicity20/0410-test/ReOriID/distmat.npy')

        # cam_distmat = np.load('./output/veri/0411-search/ReCamID/distmat.npy')
        #ori_distmat = np.load('./output/veri/ReOriID/distmat.npy')
        #distmat_np = distmat_np - 0.1 * ori_distmat# - 0.1 * cam_distmat

        indices_np = np.argsort(distmat_np, axis=1)
        if self.cfg.TEST.TRACK_RERANK and len(self.dataset.test_tracks) > 0:
            rerank_indice_by_track(indices_np, self.img_paths[self.num_query:], self.dataset.test_tracks)

        cmc, mAP = eval_func(indices_np, q_pids, g_pids, q_camids, g_camids)
        if self.cfg.TEST.WRITE_RESULT:
            np.save(os.path.dirname(self.cfg.TEST.WEIGHT) + '/distmat',
                    distmat_np)
            np.save(os.path.dirname(self.cfg.TEST.WEIGHT) + '/feats', feats.cpu().numpy())
            write_results(indices_np, os.path.dirname(self.cfg.TEST.WEIGHT),
                          self.img_paths[self.num_query:])

        return cmc, mAP, indices_np