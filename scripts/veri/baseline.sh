python tools/train.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
MODEL.POOLING_METHOD 'GeM' \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
SOLVER.LR_SCHEDULER 'warmup_cosine' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.MAX_EPOCHS 50 \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/veri/0412-search/ReCamID')"

python tools/train.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 128 \
MODEL.POOLING_METHOD 'GeM' \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
SOLVER.LR_SCHEDULER 'warmup_cosine' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.MAX_EPOCHS 50 \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/veri/0412-search/circle-N16-s128')"

python tools/train.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.25 \
SOLVER.COSINE_SCALE 64 \
MODEL.POOLING_METHOD 'GeM' \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
SOLVER.LR_SCHEDULER 'warmup_cosine' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.MAX_EPOCHS 50 \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/veri/0412-search/circle-N16-m0.25')"

