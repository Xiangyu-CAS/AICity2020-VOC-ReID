python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
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
TEST.WEIGHT "('./output/veri/0411-search/circle-N16/best.pth')"