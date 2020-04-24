from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'

# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'

_C.MODEL.MODEL_TYPE = 'baseline'
_C.MODEL.GLOBAL_DIM = 2048
_C.MODEL.LOCAL_DIM = 512
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 0.5
_C.MODEL.FROZEN_FEATURE_EPOCH = 0

_C.MODEL.FC_WEIGHT_NORM = False

_C.MODEL.POOLING_METHOD = '' # POOLING METHOD, default is avg pooling
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.DROPOUT_PROB = 0.0

_C.MODEL.EMBEDDING_HEAD = 'fc'
_C.MODEL.EMBEDDING_DIM = 512
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
_C.INPUT.COLORJIT_PROB = 1.0
_C.INPUT.AUGMIX_PROB = 0.0
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
_C.INPUT.RE_SH = 0.4
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# color space
_C.INPUT.COLOR_SPACE = 'rgb'
# random patch
_C.INPUT.RANDOM_PATCH_PROB = 0.0
# random affine
_C.INPUT.RANDOM_AFFINE_PROB = 0.0
_C.INPUT.VERTICAL_FLIP_PROB = 0.0
# random blur
_C.INPUT.RANDOM_BLUR_PROB = 0.0

# cut-off long-tailed data
_C.INPUT.CUTOFF_LONGTAILED = False
_C.INPUT.LONGTAILED_THR = 2

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.NAMES = ('market1501')
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')
# combine {train, query, gallery} in training
_C.DATASETS.COMBINEALL = False
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
# Factor of learning fc
_C.SOLVER.FC_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1

# hard example mining method of triplet loss
_C.SOLVER.HARD_EXAMPLE_MINING_METHOD = 'batch_hard'
_C.SOLVER.COSINE_MARGIN = 0.3
_C.SOLVER.COSINE_SCALE = 30
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.XBM_SIZE = 4
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# lr scheduler (warmup_multi_step, cosine)
_C.SOLVER.LR_SCHEDULER = 'warmup_multi_step'

# freeze feature epoch
_C.SOLVER.FREEZE_BASE_EPOCHS = 0

# Cyclic cosine paramter
_C.SOLVER.CYCLE_EPOCH = 30

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# support FP16
_C.SOLVER.FP16 = False

_C.SOLVER.NO_BIAS_DECAY = False
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

_C.TEST.FLIP_TEST = False

_C.TEST.DO_RERANK = False
_C.TEST.RERANK_PARAM = [20, 6, 0.3]# K1, K2, LAMBDA
_C.TEST.QUERY_EXPANSION = False
_C.TEST.DO_DBA = False
_C.TEST.TRACK_AUG = False
_C.TEST.TRACK_RERANK = False

_C.TEST.ATTRIBUTES_RERANK = False
_C.TEST.WRITE_RESULT = False
_C.TEST.USE_VOC = False
_C.TEST.ORI_DIST_PATH = ''
_C.TEST.CAM_DIST_PATH = ''
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
