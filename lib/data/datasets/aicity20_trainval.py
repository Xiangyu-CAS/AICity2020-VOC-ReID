# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET


from .bases import BaseImageDataset


class AICity20Trainval(BaseImageDataset):
    """
    将AI City train 中333个ID， 1-95为测试集, 241-478为训练集
    测试集中随机取500张作为query
    """
    dataset_dir = 'AIC20_ReID/'
    dataset_aug_dir = 'AIC20_ReID_Cropped/'
    dataset_blend_dir = 'AIC20_ReID_blend/'

    def __init__(self, root='', verbose=True, **kwargs):
        super(AICity20Trainval, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_aug_dir = osp.join(root, self.dataset_aug_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_aug_dir, 'image_train')
        self.gallery_dir = osp.join(self.dataset_aug_dir, 'image_train')
        self.train_aug_dir = osp.join(self.dataset_aug_dir, 'image_train')

        train_list_path = osp.join(self.dataset_dir, 'trainval_partial', 'train.txt')
        query_list_path = osp.join(self.dataset_dir, 'trainval_partial', 'query.txt')
        gallery_list_path = osp.join(self.dataset_dir, 'trainval_partial', 'test.txt')
        #train_aug_list_path = osp.join(self.dataset_dir, 'trainval_partial', 'train.txt')

        self._check_before_run()

        train = self._process_dir(self.train_dir, train_list_path, relabel=False)
        query = self._process_dir(self.query_dir, query_list_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, gallery_list_path, relabel=False)
        # train += self._process_dir(self.train_aug_dir, train_list_path, relabel=False)
        # train += self._process_dir(os.path.join(root, self.dataset_blend_dir, 'image_train')
        #                            , train_list_path, relabel=False)


        train = self.relabel(train)
        if verbose:
            print("=> aicity trainval loaded")
            # self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        self.train_tracks = self._read_tracks(osp.join(self.dataset_dir, 'train_track.txt'))
        self.test_tracks = self._read_tracks(osp.join(self.dataset_dir, 'trainval_partial', 'test_track.txt'))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, list_path, relabel=False):
        dataset = []
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            pid, camid, trackid, image_name = line.split('_')
            pid = int(pid)
            camid = int(camid[1:])
            img_path = osp.join(dir_path, image_name)
            dataset.append((img_path, pid, camid))
            #dataset.append((img_path, camid, pid))
        if relabel: dataset = self.relabel(dataset)

        return dataset

if __name__ == '__main__':
    dataset = AICity20Trainval(root='/home/zxy/data/ReID/vehicle')
