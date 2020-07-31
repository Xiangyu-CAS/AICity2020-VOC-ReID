python tools/train.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnet50-19c8e357.pth')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
MODEL.POOLING_METHOD 'GeM' \
SOLVER.LR_SCHEDULER 'cosine' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.MAX_EPOCHS 40 \
INPUT.SIZE_TRAIN '([256, 256])' \
INPUT.SIZE_TEST '([256, 256])' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/veri/baseline')"


python tools/train.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
MODEL.POOLING_METHOD 'GeM' \
SOLVER.LR_SCHEDULER 'cosine' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.MAX_EPOCHS 40 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/veri/0412-search/size320')"