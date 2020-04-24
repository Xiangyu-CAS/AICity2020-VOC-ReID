import torch
import numpy as np
from scipy.ndimage import label
import cv2

def extract_bbox_from_mask(input):
    assert input.ndim == 2, 'Invalid input shape'
    rows = np.any(input, axis=1)
    cols = np.any(input, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax


def localize_from_map(actmap, threshold_ratio=0.5):
    foreground_map = actmap >= (actmap.mean() * threshold_ratio)
    # single object
    try:
        bbox = extract_bbox_from_mask(foreground_map)
    except:
        bbox = None
    return bbox


def bbox_nms(bbox_list, threshold=0.5):
    bbox_list = sorted(bbox_list,  key=lambda x: x[-1], reverse=True)
    selected_bboxes = []
    while len(bbox_list) > 0:
        obj = bbox_list.pop(0)
        selected_bboxes.append(obj)
        def iou_filter(x):
            iou = compute_iou(obj[1:5], x[1:5])
            if (x[0] == obj[0] and iou >= threshold):
                return None
            else:
                return x
        bbox_list = list(filter(iou_filter, bbox_list))
    return selected_bboxes


def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(x_b - x_a + 1, 0) * max(y_b - y_a + 1, 0)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return inter_area / float(box_a_area + box_b_area - inter_area)


def draw_bbox(canvas, bboxes, color=[255, 0, 255]):
    for bbox in bboxes:
        if bbox is not None:
            cv2.rectangle(canvas, (int(bbox[0]), int(bbox[1])), ((int(bbox[2]), int(bbox[3]))), color=color)
    return canvas
