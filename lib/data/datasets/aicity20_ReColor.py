# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET
import json

from .bases import BaseImageDataset
from .aicity20 import AICity20

class AICity20ReColor(AICity20):
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
                pid = int(obj.attrib['colorID'])
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

if __name__ == '__main__':
    dataset = AICity20ReColor(root='/home/zxy/data/ReID/vehicle')


#
# # encoding: utf-8
#
# import glob
# import re
# import os
# import os.path as osp
# import xml.etree.ElementTree as ET
# import json
#
# from .bases import BaseImageDataset
#
#
# class AICity20ReOri(BaseImageDataset):
#     """
#       ----------------------------------------
#   subset   | # ids | # images | # cameras
#   ----------------------------------------
#   train    |   333 |    36935 |        36
#   query    |   333 |     1052 |        ?
#   gallery  |   333 |    18290 |        ?
#   ----------------------------------------
#
#     """
#     dataset_dir = 'AIC20_ReID/'
#     dataset_aug_dir = 'AIC20_ReID_Cropped'
#     def __init__(self, root='', verbose=True, **kwargs):
#         super(AICity20ReOri, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.dataset_aug_dir = osp.join(root, self.dataset_aug_dir)
#
#         self.train_dir = osp.join(self.dataset_aug_dir, 'image_train')
#         self.query_dir = osp.join(self.dataset_aug_dir, 'image_query')
#         self.gallery_dir = osp.join(self.dataset_aug_dir, 'image_test')
#         self.train_aug_dir = osp.join(self.dataset_aug_dir, 'image_train')
#
#         self.orientation_train_path = osp.join(self.dataset_dir, 'orientation', 'orientation_train.json')
#         self.orientation_query_path = osp.join(self.dataset_dir, 'orientation', 'orientation_query.json')
#         self.orientation_gallery_path = osp.join(self.dataset_dir, 'orientation', 'orientation_test.json')
#
#         self._check_before_run()
#
#         train = self._process_dir(self.train_dir, self.orientation_train_path, relabel=False)
#         query = self._process_dir(self.query_dir, self.orientation_query_path)
#         gallery = self._process_dir(self.gallery_dir, self.orientation_gallery_path)
#
#         #train = self.relabel(train)
#         if verbose:
#             print("=> AI CITY 2020 data loaded")
#             #self.print_dataset_statistics(train, query, gallery)
#
#         self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.train_tracks = self._read_tracks(os.path.join(self.dataset_dir, 'train_track_id.txt'))
#         self.test_tracks = self._read_tracks(os.path.join(self.dataset_dir, 'test_track_id.txt'))
#
#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
#
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#
#     def _read_orientation_info(self, path):
#         with open(path, 'r') as f:
#             orientation = json.load(f)
#         return orientation
#
#     def _process_dir(self, img_dir, json_path, relabel=False):
#         dataset = []
#         orientation_dict = self._read_orientation_info(json_path)
#         for k, v in orientation_dict.items():
#             img_path = osp.join(img_dir, k)
#             pid = int(float(v) * 360 / 10)
#             camid = 0
#             dataset.append([img_path, pid, camid])
#         if relabel: self.relabel(dataset)
#         return dataset
#
# if __name__ == '__main__':
#     dataset = AICity20ReOri(root='/home/zxy/data/ReID/vehicle')
