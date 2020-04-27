#------------------------- generate orientation-camera matrix--------------------
python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/veri/ReOriID/best.pth')"

python ./tools/aicity20/compute_distmat_from_feats.py --src_dir ./output/veri/ReOriID/



python tools/test.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.POOLING_METHOD 'GeM' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WRITE_RESULT True \
TEST.WEIGHT "('./output/veri/ReCamID/best.pth')"

python ./tools/aicity20/compute_distmat_from_feats.py --src_dir ./output/veri/ReOriID/

#------------------------- test with VOC-----------------------------------------
python tools/test.py --config_file='configs/veri.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.POOLING_METHOD 'GeM' \
DATASETS.TRAIN "('veri',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
TEST.USE_VOC False \
TEST.CAM_DIST_PATH '' \
TEST.ORI_DIST_PATH './output/veri/ReOriID/feat_distmat.npy' \
TEST.WEIGHT "('./output/veri/size320/best.pth')"