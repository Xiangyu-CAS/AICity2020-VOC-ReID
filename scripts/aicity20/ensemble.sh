python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.DO_RERANK True \
TEST.TRACK_AUG False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0408-ensemble/r50-320/best.pth')"


python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.DO_RERANK True \
TEST.TRACK_AUG False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0408-ensemble/r50/best.pth')"



python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.NAME "('resnet101_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.DO_RERANK True \
TEST.TRACK_AUG False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0408-ensemble/r101/best.pth')"


python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.NAME "('se_resnet101_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.DO_RERANK True \
TEST.TRACK_AUG False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0408-ensemble/se-r101/best.pth')"