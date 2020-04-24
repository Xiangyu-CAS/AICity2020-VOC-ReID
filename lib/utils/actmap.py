import numpy as np
import torch

'''
input: featmap: torch.tensor.float(), N*C*H*W 
output: foreground_mask, background_mask: torch.tensor.bool(), N*1*H*W
'''
def batch_attention_mask(featmap, threshold=0.75):
    actmap = (featmap**2).sum(dim=1) # N*W*H
    val = actmap.view(actmap.size(0), -1)
    min_val, _ = val.min(dim=1)
    max_val, _ = val.max(dim=1)
    thr = min_val + (max_val - min_val) * threshold
    for i in range(actmap.size(0)):
        actmap[i] = actmap[i] < thr[i]
    return actmap.unsqueeze(dim=1)


def generate_attention_mask(actmap, threshold=1.0):
    actmap = (actmap - np.min(actmap)) / (
            np.max(actmap) - np.min(actmap) + 1e-12
    )
    foreground_mask = actmap >= (actmap.mean() * threshold)
    background_mask = actmap < (actmap.mean() * threshold)
    return foreground_mask, background_mask
