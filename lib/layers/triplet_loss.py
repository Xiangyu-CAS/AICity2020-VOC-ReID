# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, mining_method='batch_hard', return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    if mining_method == 'batch_hard':
        # dist_ap, dist_an = [], []
        # for i in range(N):
        #     dist_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
        #     dist_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
        # dist_ap = torch.cat(dist_ap).unsqueeze(dim=1)
        # dist_an = torch.cat(dist_an).unsqueeze(dim=1)

        # Batch Hard
        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
    elif mining_method == 'batch_sample':
        dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
        relative_p_inds = torch.multinomial(
            F.softmax(dist_mat_ap, dim=1), num_samples=1)
        dist_ap = torch.gather(dist_mat_ap, 1, relative_p_inds)

        dist_mat_an = dist_mat[is_neg].contiguous().view(N, -1)
        relative_n_inds = torch.multinomial(
            F.softmin(dist_mat_an, dim=1), num_samples=1)
        dist_an = torch.gather(dist_mat_an, 1, relative_n_inds)
    elif mining_method == 'batch_soft':
        dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
        dist_mat_an = dist_mat[is_neg].contiguous().view(N, -1)
        weight_ap = torch.exp(dist_mat_ap) / torch.exp(dist_mat_ap).sum(dim=1, keepdim=True)
        weight_an = torch.exp(-dist_mat_an) / torch.exp(-dist_mat_an).sum(dim=1, keepdim=True)

        dist_ap = (weight_ap * dist_mat_ap).sum(dim=1, keepdim=True)
        dist_an = (weight_an * dist_mat_an).sum(dim=1, keepdim=True)
    else:
        print("error, unsupported mining method {}".format(mining_method))

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, mining_method='batch_hard'):
        self.margin = margin
        self.mining_method = mining_method
        if margin > 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            # global_feat = normalize(global_feat, axis=-1)
            global_feat = torch.nn.functional.normalize(global_feat, dim=1, p=2)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, self.mining_method)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin > 0:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        # loss += 0.2*(dist_ap.mean(0)/dist_an.mean(0))
        return loss


class CenterTripletLoss(object):
    def __init__(self, margin=None):
        self.margin = margin
        if margin > 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feats, labels):
        N = feats.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        avg_feats = []
        for i in range(N):
            avg_feat = feats[is_pos[i], :].mean(dim=0)
            avg_feats.append(avg_feat)
        avg_feats = torch.stack(avg_feats, dim=0)
        dist_mat = euclidean_dist(feats, avg_feats)
        is_pos = torch.eye(N, device=feats.device).bool()
        is_neg = (1 - torch.eye(N, device=feats.device)).bool()
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1)
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin > 0:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=log_probs.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss