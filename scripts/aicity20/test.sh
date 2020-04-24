#----------------------- 50-circle------------------------------------------
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
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.WRITE_RESULT True \
TEST.USE_VOC True \
TEST.CAM_DIST_PATH './output/aicity20/0409-ensemble/ReCamID/feat_distmat.npy' \
TEST.ORI_DIST_PATH './output/aicity20/0409-ensemble/ReOriID/feat_distmat.npy' \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/r50-320-circle/best.pth')"

#------------------------101-circle-----------------------------------------
python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.NAME "('resnet101_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.DO_RERANK True \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.WRITE_RESULT True \
TEST.USE_VOC True \
TEST.CAM_DIST_PATH './output/aicity20/0409-ensemble/ReCamID/feat_distmat.npy' \
TEST.ORI_DIST_PATH './output/aicity20/0409-ensemble/ReOriID/feat_distmat.npy' \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/r101-320-circle/best.pth')"

#----------------------- next-cirlce ---------------------------------------
python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.NAME "('resnext101_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.DO_RERANK True \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.WRITE_RESULT True \
TEST.USE_VOC True \
TEST.CAM_DIST_PATH './output/aicity20/0409-ensemble/ReCamID/feat_distmat.npy' \
TEST.ORI_DIST_PATH './output/aicity20/0409-ensemble/ReOriID/feat_distmat.npy' \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/next101-320-circle/best.pth')"