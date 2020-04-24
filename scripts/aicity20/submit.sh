#--------------------server submit------------------------
python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.QUERY_EXPANSION True \
TEST.DO_RERANK False \
TEST.TRACK_AUG False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0408-ensemble/r50-320/best.pth')"
# TEST.RERANK_PARAM "([50, 15, 0.5])" \

python tools/aicity20/submit.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('7')" \
MODEL.NAME "('resnet101_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.DO_RERANK False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST True \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/r101-320-after/best.pth')"



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
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/next101-320-circle/best.pth')"

#--------------------server test---------------------------
# "('aicity20-trainval',)"

python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('7')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20-test',)" \
DATASETS.TEST "('aicity20-test',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.QUERY_EXPANSION False \
TEST.DO_DBA False \
TEST.TRACK_AUG False \
TEST.DO_RERANK True \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST False \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-search/r50-triplet-after/best.pth')"
#
#INPUT.SIZE_TRAIN '([320, 320])' \
#INPUT.SIZE_TEST '([320, 320])' \

# ReCamID
python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.QUERY_EXPANSION False \
TEST.DO_DBA False \
TEST.TRACK_AUG False \
TEST.DO_RERANK False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST False \
TEST.TRACK_RERANK False \

TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0407-ReCamID/best.pth')"


# ReOriID
python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.QUERY_EXPANSION False \
TEST.DO_DBA False \
TEST.TRACK_AUG False \
TEST.DO_RERANK False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST False \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/ReOriID/best.pth')"
#TEST.WEIGHT "('./output/aicity20/0409-ReOriID/best.pth')"


python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.QUERY_EXPANSION False \
TEST.DO_DBA False \
TEST.TRACK_AUG False \
TEST.DO_RERANK False \
TEST.RERANK_PARAM "([50, 15, 0.5])" \
TEST.FLIP_TEST False \
TEST.TRACK_RERANK False \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/ReTypeID/best.pth')"