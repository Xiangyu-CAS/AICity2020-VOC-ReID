#----------------------- ensemble three models (resnet50, resnet101, resnext101)-------------------------------
python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
DATASETS.TRAIN "('aicity20', 'aicity20-sim',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0409-ensemble/r50-320-circle')"


python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
DATASETS.TRAIN "('aicity20', 'aicity20-sim',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0409-ensemble/r101-320-circle')"


python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnext101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnext101_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
DATASETS.TRAIN "('aicity20', 'aicity20-sim',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0409-ensemble/next101-320-circle')"
