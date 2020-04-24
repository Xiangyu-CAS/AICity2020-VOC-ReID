python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('regnety_800mf')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/RegNetY-800MF.pth')" \
SOLVER.LR_SCHEDULER 'cosine' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 10 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
DATASETS.TRAIN "('aicity20-trainval',)" \
DATASETS.TEST "('aicity20-trainval',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0420-search/regnet')"


python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnest50')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnest50-528c19ca.pth')" \
SOLVER.LR_SCHEDULER 'cosine' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 10 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
DATASETS.TRAIN "('aicity20-trainval',)" \
DATASETS.TEST "('aicity20-trainval',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0420-search/nest50')"



python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnest50')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnest50-528c19ca.pth')" \
SOLVER.LR_SCHEDULER 'cosine' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 10 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
DATASETS.TRAIN "('aicity20-trainval',)" \
DATASETS.TEST "('aicity20-trainval',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0417-search/nest50-ibn')"