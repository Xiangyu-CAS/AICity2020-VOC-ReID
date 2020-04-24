# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET


from .bases import BaseImageDataset
from .aicity20 import AICity20

class AICity20Sim(AICity20):
    """
    Simulation data: include attribute information
    - orientation
    - color
    - cls type (truck, suv)
    """
    dataset_dir = 'AIC20_ReID_Simulation'
    def __init__(self, root='', verbose=True, **kwargs):
        super(AICity20, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.list_train_path = osp.join(self.dataset_dir, 'name_train.txt')
        self.train_label_path = osp.join(self.dataset_dir, 'train_label.xml')
        self._check_before_run()

        train = self._process_dir(self.train_dir, self.list_train_path, self.train_label_path, relabel=True)

        if verbose:
            print("=> AI CITY 2020 sim data loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = []
        self.gallery = []

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


if __name__ == '__main__':
    dataset = AICity20(root='/home/zxy/data/ReID/vehicle')
