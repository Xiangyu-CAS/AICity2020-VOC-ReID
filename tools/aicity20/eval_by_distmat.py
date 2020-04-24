import numpy as np
import os
import sys
import torch

sys.path.append('.')
from lib.data.datasets.veri import VeRi
from lib.data.datasets.aicity20_trainval import AICity20Trainval
from lib.utils.post_process import build_track_lookup, re_ranking


def generate_track_results(distmat, tracks, topk=100):
    indice = np.argsort(distmat, axis=1)
    results = []
    m, n =distmat.shape
    for i in range(m):
        result = []
        track_idxs = indice[i]
        for idx in track_idxs:
            result.extend(tracks[idx])
        results.append(result[:topk])
    return results


def results_to_pid(results, img_to_pid):

    result_pids = []
    for line in results:
        result_pid = []
        for name in line:
            result_pid.append(img_to_pid[name])
        result_pids.append(result_pid)
    return result_pids


def eval_results(query_pids, gallery_pids, result_pids):
    query_pids = np.array(query_pids)
    gallery_pids = np.array(gallery_pids)
    result_pids = np.array(result_pids)
    gt_match = gallery_pids == query_pids[:, np.newaxis]

    all_cmc = []
    all_AP = []
    num_valid_q = 0
    for i in range(len(query_pids)):
        if not np.any(gt_match[i]):
            continue
        num_valid_q += 1
        num_rel = gt_match[i].sum()
        match = query_pids[i] == result_pids[i]
        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        tmp_cmc = match.cumsum()
        tmp_cmc = np.array(tmp_cmc) / (np.arange(len(tmp_cmc)) + 1.)
        tmp_cmc = np.asarray(tmp_cmc) * match
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.array(all_cmc).sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc[r - 1]))


def generate_results(distmat, gallery, topk=100):
    assert distmat.shape[1] == len(gallery)
    names = [os.path.basename(img_path) for img_path, pid, camid in gallery]
    indice = np.argsort(distmat, axis=1)
    indice = indice[:, :topk]
    results = []
    m, n = indice.shape
    for i in range(m):
        result = []
        for j in range(n):
            result.append(names[indice[i, j]])
        results.append(result)
    return results


def results_to_track(results, tracks, topk=100):
    m, n = len(results), len(results[0])
    lookup_map = {}
    for i, track in enumerate(tracks):
        for img_id in track:
            lookup_map[img_id] = i
    reranked_results = []
    for i in range(m):
        used_track_id = set()
        reranked_result = []
        for j in range(n):
            track_id = lookup_map[results[i][j]]
            if track_id in used_track_id:
                continue
            used_track_id.add(track_id)
            reranked_result.extend(tracks[track_id])
        reranked_results.append(reranked_result[:topk])
    return reranked_results



if __name__ == '__main__':
    dataset = VeRi(root='/home/zxy/data/ReID/vehicle')
    distmat1 = np.load('./output/veri/0411-search/circle-N16/distmat.npy')
    cam_distmat = np.load('./output/veri/0411-search/ReCamID/distmat.npy')
    ori_distmat = np.load('./output/veri/0411-search/ReOriID/distmat.npy')

    # type_distmat = np.load('./output/aicity20/0409-ReTypeID/feat_distmat.npy')
    # color_distmat = np.load('./output/aicity20/0409-ReColorID/feat_distmat.npy')

    # cam_distmat = np.load('./output/aicity20/experiments/ReCamID/distmat.npy')
    # ori_distmat = np.load('./output/aicity20/experiments/ReOriID/distmat.npy')

    #cam_distmat = np.load('./output/aicity20/0407-ReCamID/distmat_test.npy')
    #ori_distmat = np.load('./output/aicity20/0409-ReOriID/distmat.npy')
    # distmat5 = np.load('./output/aicity20/0407-ensemble/se-r101/distmat.npy')

    qf = torch.rand(len(dataset.query), 1)
    gf = torch.rand(len(dataset.gallery), 1)

    distmat = distmat1- 0.1 * ori_distmat - 0.1 * cam_distmat
    #distmat = distmat[:len(qf), len(qf):]
    #distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.5, local_distmat=distmat, only_local=True)


    query_pids = [pid for _, pid, _ in dataset.query]
    gallery_pids = []
    img_to_pid = {}
    for img_path, pid, _ in dataset.gallery:
        name = os.path.basename(img_path)
        gallery_pids.append(pid)
        img_to_pid[name] = pid

    #distmat = (distmat1 + distmat2 + distmat3 + distmat5) / 4 #- 0.1 * distmat4
    #distmat = distmat1 * distmat2 * distmat3 * distmat5
    results = generate_results(distmat, dataset.gallery, topk=-1)
    #results = results_to_track(results, dataset.test_tracks)
    result_pids = results_to_pid(results, img_to_pid)
    eval_results(query_pids, gallery_pids, result_pids)
