import os
import cv2
import numpy as np

def concat_vis(imgs, size = [256, 128]):
    n = len(imgs)
    canvas = np.zeros((size[0], n * size[1], 3)) #(h*w*c)
    for i, img in enumerate(imgs):
        img = cv2.resize(img, (size[1], size[0])) # (w*h)
        canvas[:, i*size[1]:(i+1)*size[1], :] = img
    return canvas