# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset, BaseImageDataset, apply_id_bias
from .samplers import RandomIdentitySampler, MPerClassSampler  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg, shuffle_train=True):
    train_transforms = build_transforms(cfg, is_train=shuffle_train)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = BaseImageDataset()
    # LOAD TRAIN
    print(cfg.DATASETS.TRAIN)
    if isinstance(cfg.DATASETS.TRAIN, str):
        cur_dataset = init_dataset(cfg.DATASETS.TRAIN, root=cfg.DATASETS.ROOT_DIR)
        dataset = cur_dataset
    else:
        for i, dataset_name in enumerate(cfg.DATASETS.TRAIN):
            cur_dataset = init_dataset(dataset_name, root=cfg.DATASETS.ROOT_DIR)
            min_id, max_id = dataset.get_id_range(dataset.train)
            dataset.train.extend(apply_id_bias(cur_dataset.train, id_bias=max_id + 1))
            dataset.train_tracks += cur_dataset.train_tracks
            if cfg.DATASETS.COMBINEALL:
                min_id, max_id = dataset.get_id_range(dataset.train)
                to_merge_train = dataset.relabel(cur_dataset.query + cur_dataset.gallery)
                dataset.train.extend(apply_id_bias(to_merge_train, id_bias=max_id + 1))
                dataset.train_tracks += cur_dataset.test_tracks
        dataset.train = dataset.relabel(dataset.train) # in case of inconsistent ids

    # cutoff long tailed data
    if cfg.INPUT.CUTOFF_LONGTAILED:
        dataset.train = dataset.longtail_data_process(dataset.train,
                                                      NUM_INSTANCE_PER_CLS=cfg.INPUT.LONGTAILED_THR)

    # LOAD VALIDATE
    if isinstance(cfg.DATASETS.TEST, str):
        cur_dataset = init_dataset(cfg.DATASETS.TEST, root=cfg.DATASETS.ROOT_DIR)
        dataset.query, dataset.gallery = cur_dataset.query, cur_dataset.gallery
        dataset.test_tracks = cur_dataset.test_tracks
        dataset.query_orientation = cur_dataset.query_orientation
        dataset.gallery_orientation = cur_dataset.gallery_orientation
    else:
        dataset.query, dataset.gallery = [], []
        for i, dataset_name in enumerate(cfg.DATASETS.TEST):
            cur_dataset = init_dataset(dataset_name, root=cfg.DATASETS.ROOT_DIR)
            dataset.query.extend(apply_id_bias(cur_dataset.query, id_bias=i * 10000))
            dataset.gallery.extend(apply_id_bias(cur_dataset.gallery, id_bias=i * 10000))
            dataset.test_tracks += cur_dataset.test_tracks
            dataset.query_orientation = cur_dataset.query_orientation
            dataset.gallery_orientation = cur_dataset.gallery_orientation
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)
    num_train_pids, num_train_imgs, num_train_cams = dataset.get_imagedata_info(dataset.train)

    num_classes = num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=shuffle_train, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'm_per_class':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=MPerClassSampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes, dataset
