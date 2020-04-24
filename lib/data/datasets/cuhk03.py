# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""
import os
import glob
import re
import os.path as osp
from scipy.io import loadmat

from lib.utils.iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseImageDataset


class CUHK03(BaseImageDataset):
    """
    CUHK03
    Reference:
    Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    URL: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!

    Dataset statistics:
    # identities: 1360
    # images: 13164
    # cameras: 6
    # splits: 20 (classic)
    Args:
        split_id (int): split index (default: 0)
        cuhk03_labeled (bool): whether to load labeled images; if false, detected images are loaded (default: False)
    """
    dataset_dir = 'cuhk03'

    def __init__(self, root='', cuhk03_labeled=False, verbose=True,
                 **kwargs):
        super(CUHK03, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')
        self._check_before_run()

        if cuhk03_labeled:
            image_type = 'cuhk03_labeled'
        else:
            image_type = 'cuhk03_detected'
        self.dataset_dir = osp.join(self.dataset_dir, image_type)

        train = self.process_dir(self.dataset_dir, relabel=True)
        query = []
        gallery = []

        if verbose:
            print("=> CUHK03 ({}) loaded".format(image_type))
            # self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def process_dir(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pid_container = set()
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            video, pid, camid, _ = img_name.split('_')
            video, pid, camid = int(video), int(pid), int(camid)
            pid = (video-1) * 1000 + pid
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            video, pid, camid, _ = img_name.split('_')
            video, pid, camid = int(video), int(pid), int(camid)
            pid = (video-1) * 1000 + pid
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

if __name__ == '__main__':
    dataset = CUHK03(root='/home/zxy/data/ReID')