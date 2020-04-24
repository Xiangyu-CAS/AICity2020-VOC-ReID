import numpy as np
import cv2
import os
import sys

sys.path.append('.')
from lib.data.datasets.aicity20_trainval import AICity20Trainval

def visualize_submit(dataset, out_dir, submit_txt_path, topk=5):
    query_dir = dataset.query_dir
    gallery_dir = dataset.gallery_dir

    vis_size = (256, 256)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    results = []
    with open(submit_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            results.append(line.split(' '))

    query_pids = [pid for _, pid, _ in dataset.query]
    img_to_pid = {}
    for img_path, pid, _ in dataset.gallery:
        name = os.path.basename(img_path)
        img_to_pid[name] = pid

    for i, result in enumerate(results):
        is_False = False
        # query_path = os.path.join(query_dir, str(i+1).zfill(6)+'.jpg')
        query_path = os.path.join(query_dir, os.path.basename(dataset.query[i][0]))
        gallery_paths = []
        for name in result:
            # gallery_paths.append(os.path.join(gallery_dir, index.zfill(6)+'.jpg'))
            gallery_paths.append(os.path.join(gallery_dir, name))

        imgs = []
        imgs.append(cv2.resize(cv2.imread(query_path), vis_size))
        for n in range(topk):
            img = cv2.resize(cv2.imread(gallery_paths[n]), vis_size)
            if query_pids[i] != img_to_pid[result[n]]:
                img = cv2.rectangle(img, (0, 0), vis_size, (0, 0, 255), 2)
                is_False = True
            imgs.append(img)

        canvas = np.concatenate(imgs, axis=1)
        #if is_False:
        cv2.imwrite(os.path.join(out_dir, os.path.basename(query_path)), canvas)


if __name__ == '__main__':
    # dataset_dir = '/home/xiangyuzhu/data/ReID/AIC20_ReID'
    dataset = AICity20Trainval(root='/home/zxy/data/ReID/vehicle')
    #
    # dataset_dir = '/home/zxy/data/ReID/vehicle/AIC20_ReID_Cropped'
    # query_dir = os.path.join(dataset_dir, 'image_query')
    # gallery_dir = os.path.join(dataset_dir, 'image_test')

    out_dir = 'vis/'
    submit_txt_path = './output/aicity20/experiments/circle-sim-aug/result_voc.txt'
    visualize_submit(dataset, out_dir, submit_txt_path)
