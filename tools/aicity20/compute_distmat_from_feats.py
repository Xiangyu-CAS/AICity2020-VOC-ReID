import torch
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--src_dir", default="./output/aicity20/0410-test/r50-320-circle", help="path to config file", type=str
    )
    args = parser.parse_args()
    src_dir = args.src_dir

    feat = np.load(src_dir + '/' + 'feats.npy')
    feat = torch.tensor(feat, device='cuda')
    all_num = len(feat)
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
              torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    distmat.addmm_(1, -2, feat, feat.t())
    distmat = distmat.cpu().numpy()
    np.save(src_dir + '/' + 'feat_distmat', distmat)
