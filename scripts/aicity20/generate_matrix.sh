# ReCamID
python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/ReCamID/best.pth')"

python ./tools/aicity20/compute_distmat_from_feats.py --src_dir ./output/aicity20/0409-ensemble/ReCamID/

# ReOriID
python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/aicity20/0409-ensemble/ReOriID/best.pth')"

python ./tools/aicity20/compute_distmat_from_feats.py --src_dir ./output/aicity20/0409-ensemble/ReOriID/