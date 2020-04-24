# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET
import json

from .bases import BaseImageDataset
from .aicity20 import AICity20

class AICity20ReType(AICity20):
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

        train = self._process_dir(self.train_dir, self.list_train_path, self.train_label_path, relabel=False)

        train_num = 180000
        #train_num = 100000
        #train_num = 50000
        query_num = 500
        gallery_num = 5000
        query = train[train_num:train_num+query_num]
        gallery = train[train_num+query_num: train_num+query_num+gallery_num]
        train = train[:train_num]

        if verbose:
            print("=> AI CITY 2020 sim data loaded")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, img_dir, list_path, label_path, relabel=False):
        dataset = []
        if label_path:
            tree = ET.parse(label_path, parser=ET.XMLParser(encoding='utf-8'))
            objs = tree.find('Items')
            for obj in objs:
                image_name = obj.attrib['imageName']
                img_path = osp.join(img_dir, image_name)
                pid = int(obj.attrib['typeID'])
                camid = int(obj.attrib['cameraID'][1:])
                dataset.append((img_path, pid, camid))
            if relabel: dataset = self.relabel(dataset)
        else:
            with open(list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    img_path = osp.join(img_dir, line)
                    pid = 0
                    camid = 0
                    dataset.append((img_path, pid, camid))
        return dataset


