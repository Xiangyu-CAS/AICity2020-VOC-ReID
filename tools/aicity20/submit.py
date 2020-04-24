# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.inference import inference, select_topk
from lib.modeling import build_model
from lib.utils.logger import setup_logger
from lib.data.datasets.aicity20 import AICity20

def write_result(indices, dst_dir, topk=100):
    indices = indices[:, :topk]
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))
    with open(os.path.join(dst_dir, 'track2.txt'), 'w') as f:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f.write(write_line)

'''
根据rank K, 选择整个track，直到把topk填满
'''
def write_result_with_track(indices, dst_dir, tracks, topk=100):
    indices = indices[:, :topk]
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))

    results = []
    for i in range(m):
        results.append((indices[i] + 1).tolist())

    # rerank results according to tracks
    lookup_map = {}
    for i, track in enumerate(tracks):
        for img_id in track:
            lookup_map[int(img_id)] = i
        # for img_name in track:
        #     lookup_map[int(img_name.split('.')[0])] = i
    reranked_results = []
    for i in range(m):
        used_track_id = set()
        reranked_result = []
        for j in range(topk):
            track_id = lookup_map[results[i][j]]
            if track_id in used_track_id:
                continue
            used_track_id.add(track_id)
            reranked_result.extend(tracks[track_id])
            if len(reranked_result) >= topk: break
        reranked_results.append(reranked_result[:topk])

    with open(os.path.join(dst_dir, 'track2.txt'), 'w') as f:
        for i in range(m):
            write_line = ' '.join(map(str, reranked_results[i])) + '\n'
            f.write(write_line)


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="./configs/debug.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    indices_np = inference(cfg, model, val_loader, num_query, dataset)

    ## read meta information
    dataset = AICity20(cfg.DATASETS.ROOT_DIR)
    #write_result(indices_np, os.path.dirname(cfg.TEST.WEIGHT), topk=100)
    write_result_with_track(indices_np, os.path.dirname(cfg.TEST.WEIGHT), dataset.test_tracks)

if __name__ == '__main__':
    main()
