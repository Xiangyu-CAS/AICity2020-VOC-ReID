python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_TYPE 'triplet' \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
MODEL.IF_LABELSMOOTH 'off' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
DATASETS.TRAIN "('aicity20-ReCam',)" \
DATASETS.TEST "('aicity20-ReCam',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0409-ReCamID')"