# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, cams, tracklet_stats = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """
    def __init__(self):
        self.train = []
        self.query = []
        self.gallery = []
        self.train_tracks = [] # track information
        self.test_tracks = []
        self.query_orientation = None
        self.gallery_orientation = None

    def longtail_data_process(self, data, NUM_INSTANCE_PER_CLS=2):
        labels = {}
        for img_path, pid, camid in data:
            if pid in labels:
                labels[pid].append([img_path, pid, camid])
            else:
                labels[pid] = [[img_path, pid, camid]]

        # cut-off long-tail data
        keep_data = []
        remove_data = []
        for key, value in labels.items():
            if len(value) < NUM_INSTANCE_PER_CLS:
                remove_data.extend(value)
                continue
            keep_data.extend(value)
        keep_data = self.relabel(keep_data)

        # import shutil
        # import os
        # dst_dir = './longtailed-N3'
        # for img_path, pid, camid in remove_data:
        #     dst_path = os.path.join(dst_dir, str(pid).zfill(5))
        #     if not os.path.exists(dst_path):
        #         os.makedirs(dst_path)
        #     shutil.copyfile(img_path, os.path.join(dst_path, os.path.basename(img_path)))

        return keep_data

    def combine_all(self):
        # combine train, query, gallery
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(self.train)
        new_train = self.query + self.gallery
        #new_train = self.relabel(new_train)

        for img_path, pid, camid in new_train:
            self.train.append([img_path, pid + num_train_pids, camid])
        self.train = self.relabel(self.train)
        self.query = []
        self.gallery = []

    def get_id_range(self, lists):
        pid_container = set()
        for img_path, pid, camid in lists:
            pid_container.add(pid)

        if len(pid_container) == 0:
            min_id, max_id = 0, 0
        else:
            min_id, max_id = min(pid_container), max(pid_container)
        return min_id, max_id

    def relabel(self, lists):
        relabeled = []
        pid_container = set()
        for img_path, pid, camid in lists:
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path, pid, camid in lists:
            pid = pid2label[pid]
            relabeled.append([img_path, pid, camid])
        return relabeled

    def _read_tracks(self, path):
        tracks = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                track = line.split(' ')
                tracks.append(track)
        return tracks

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class BaseVideoDataset(BaseDataset):
    """
    Base class of video reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(train, return_tracklet_stats=True)

        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(query, return_tracklet_stats=True)

        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")


def apply_id_bias(train, id_bias=0):
    # add id bias
    id_biased_train = []
    for img_path, pid, camid in train:
        id_biased_train.append([img_path, pid + id_bias, camid])
    return id_biased_train
