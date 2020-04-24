#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09

@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""
import torch.nn.functional as F
import numpy as np
import torch
import os
from sklearn.decomposition import PCA


def comput_distmat(qf, gf, input_type='torch'):
    m, n = qf.shape[0], gf.shape[0]
    if input_type == 'numpy':
        # TODO : using numpy to compute distmat
        pass
    else:
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        indices = torch.argsort(distmat, dim=1)
    return distmat, indices

# DBA
def database_aug(gf, top_k=10):
    # distmat, indices = comput_distmat(gf, gf)
    m, n = gf.shape[0], gf.shape[0]
    distmat = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, gf, gf.t())
    indices = np.argsort(distmat.cpu().numpy(), axis=1)
    expanded_gf = (gf[indices[:, :top_k]]).mean(dim=1)
    return expanded_gf

# QE
def average_query_expansion(qf, feats, top_k=6):
    _, indices = comput_distmat(qf, feats)
    expanded_qf = (feats[indices[:, :top_k]]).mean(dim=1)
    return expanded_qf


def alpha_query_expansion(qf, feats, alpha=3.0, top_k=10):
    # qf : m x N, gf: n x N
    # distmat : m x n, indices: m x n
    # weights: m x topk
    distmat, indices = comput_distmat(qf, feats)
    # print(distmat.shape, indices.shape)
    # print(indices[:, :top_k].shape)
    # weights = torch.pow(distmat[indices[:, :top_k]], alpha) # m x topk
    # print(weights.shape)
    expanded_qf = (feats[indices[:, :top_k]]).mean(dim=1)
    return expanded_qf

# rerank
def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False,
               USE_VOC=False, cam_dist=None, ori_dist=None):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat

    # cam_dist = np.load('./output/aicity20/0409-ensemble/ReCamID/feat_distmat.npy')
    # ori_dist = np.load('./output/aicity20/0409-ensemble/ReOriID/feat_distmat.npy')
    if USE_VOC:
        original_dist = original_dist - 0.1 * ori_dist - 0.1 * cam_dist # - 0.04 * type_dist

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def build_track_lookup(tracks):
    lookup = {} # {img_name: track_id}
    for i, track in enumerate(tracks):
        for img_name in track:
            lookup[img_name] = i
    return lookup

def track_aug(feats, tracks, img_paths):
    assert len(feats) == len(img_paths), 'len(feats) != len(img_paths)'
    lookup = {} # {image_name: track_id}
    print("track={}".format(len(tracks)))
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    for i, track in enumerate(tracks):
        for img_name in track:
            lookup[img_name] = i

    average_seq = [[] for i in range(len(tracks))]
    for idx, img_name in enumerate(img_names):
        if img_name in lookup:
            average_seq[lookup[img_name]].append(idx)

    for seq in average_seq:
        if len(seq) == 0: continue
        # average
        # avg_feat = feats[seq, :].mean(dim=0)
        # #avg_feat, _ = feats[seq, :].max(dim=0)
        # feats[seq, :] = avg_feat

        # weighted average
        track_feats = feats[seq, :].clone() # N*DIM
        dist, indices = comput_distmat(track_feats, track_feats) # N*N
        weights = 1 / (dist + 0.01) # N*N
        #weights = torch.exp(-dist) / torch.exp(-dist).sum()
        feats[seq, :] = F.linear(weights.t(), track_feats.t())
        for i, idx in enumerate(seq):
            weight = weights[i].unsqueeze(dim=1).expand_as(track_feats)
            feats[idx, :] = (weight * track_feats).mean(dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats


def rerank_indice_by_track(indices, img_paths, tracks):
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    assert indices.shape[1] == len(img_names), 'wrong'
    name_to_trackid = build_track_lookup(tracks) # {img_name: track_id}
    name_to_idx = {img_name: i for i, img_name in enumerate(img_names)}

    track_indices = []
    for i in range(indices.shape[0]):
        used_track = set()
        indice = indices[i]
        track_indice = []
        for idx in indice:
            img_name = img_names[idx]
            track_id = name_to_trackid[img_name]
            if track_id in used_track: continue
            used_track.add(track_id)
            track_indice.append(track_id)
        track_indices.append(track_indice)
        # rewrite indices by track
        count = 0
        for track_id in track_indice:
            names = tracks[track_id]
            idxs = []
            for name in names:
                if name not in name_to_idx: continue
                idxs.append(name_to_idx[name])
                #idxs = [name_to_idx[name] for name in names]
            indices[i][count:count+len(idxs)] = idxs
            count += len(idxs)
    return indices


def pca_whiten(qf, gf, dim=256):
    qf, gf = qf.cpu().numpy(), gf.cpu().numpy()
    pca = PCA(n_components=dim, whiten=False)
    gf_new = pca.fit_transform(gf)
    qf_new = pca.transform(qf)
    return torch.tensor(qf_new, device='cuda'), torch.tensor(gf_new, device='cuda')


def encode_gf2tf(gf, img_paths, tracks):
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    assert gf.shape[0] == len(img_names), 'wrong'
    name_to_trackid = build_track_lookup(tracks) # {img_name: track_id}
    name_to_idx = {img_name: i for i, img_name in enumerate(img_names)}

    tf = torch.zeros((len(tracks), gf.size(1)), device=gf.device)
    for i, track in enumerate(tracks):
        idxs = []
        for name in track:
            if name not in name_to_idx: continue
            idxs.append(name_to_idx[name])
        tf[i, :], _ = gf[idxs, :].mean(dim=0)
    return tf

def decode_tf2gf(tf, img_paths, tracks):
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    assert tf.shape[0] == len(tracks), 'wrong'
    name_to_trackid = build_track_lookup(tracks) # {img_name: track_id}
    name_to_idx = {img_name: i for i, img_name in enumerate(img_names)}

    gf = torch.rand((len(img_paths), tf.size(1)), device=tf.device)
    for i in range(len(tf)):
        idxs = []
        for name in tracks[i]:
            if name not in name_to_idx: continue
            idxs.append(name_to_idx[name])
        gf[idxs, :] = tf[i, :]
    return gf


def decode_trackIndice(trackIndice, img_paths, tracks):
    assert trackIndice.shape[1] == len(tracks), 'wrong'
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    name_to_trackid = build_track_lookup(tracks) # {img_name: track_id}
    name_to_idx = {img_name: i for i, img_name in enumerate(img_names)}
    indices = np.zeros((trackIndice.shape[0], len(img_paths)))
    for i in range(trackIndice.shape[0]):
        track_indice = trackIndice[i]
        count = 0
        for track_id in track_indice:
            names = tracks[track_id]
            idxs = []
            for name in names:
                if name not in name_to_idx: continue
                idxs.append(name_to_idx[name])
            indices[i, count:count+len(idxs)] = idxs
            count += len(idxs)
    return indices.astype(np.int)


def orientation_penalize(distmat, img_paths, dataset, weight=0.1):
    m, n = distmat.shape
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    query_names = img_names[:m]
    gallery_names = img_names[m:]

    query_oris = [float(dataset.query_orientation[name]) for name in query_names]
    gallery_oris = [float(dataset.gallery_orientation[name]) for name in gallery_names]

    query_oris = torch.tensor(query_oris, device='cuda')
    gallery_oris = torch.tensor(gallery_oris, device='cuda')

    oris_dist, _ = comput_distmat(query_oris.unsqueeze(dim=1), gallery_oris.unsqueeze(dim=1))
    oris_dist = torch.nn.functional.normalize(oris_dist, dim=1, p=2)
    print(distmat[:10, :10])
    print(oris_dist[:10, :10])
    distmat += weight * ((360 - oris_dist) / 360)
    return distmat

def write_results(indices, out_dir, gallery_paths, topk=100):
    gallery_names = [os.path.basename(path) for path in gallery_paths]
    indices = indices[:, :topk]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))
    with open(os.path.join(out_dir, 'result.txt'), 'w') as f:
        for i in range(m):
            write_line = [gallery_names[indices[i, j]] for j in range(n)]
            write_line = ' '.join(map(str, write_line)) + '\n'
            f.write(write_line)


def generate_track_idxs(gallery_names, tracks):
    track_idxs = []
    img_to_idx = {name:i for i, name in enumerate(gallery_names)}
    for track in tracks:
        idxs = []
        for name in track:
            if name not in img_to_idx: continue
            idxs.append(img_to_idx[name])
        track_idxs.append(idxs)
    return track_idxs

def generate_track_distmat(distmat, track_idxs):

    track_distmat = []
    for i, track_idx in enumerate(track_idxs):
        track_distmat.append(distmat[:, track_idx].min(1)[:, np.newaxis])
    track_distmat = np.hstack(track_distmat)
    return track_distmat
