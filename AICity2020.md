### Prepare Data
- Download [CityFlow](https://www.aicitychallenge.org/) and [VehicleX](https://github.com/yorkeyao/VehicleX)
, rename them to 'AIC20_ReID' and 'AIC20_ReID_Simulation' respectively.
- Weakly Supervised crop Augmentation. Crop vehicle in image via weakly supervised method, 
a vehicle ReID pretrain model is needed to generate attention map. If you dont want to train
it yourself you can get [pretrain model](https://drive.google.com/drive/folders/1cKMcG9g4FooLMI8CLfsK-lmOzTEwijtS?usp=sharing)

````
# first temporary comment aicity20.py line 49 # train += self._process_dir(self.train_aug_dir, self.list_train_path, self.train_label_path, relabel=False)
# to make sure only original data is used
# step 1: train inital model
python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('./output/aicity20/0326-search/augmix/')"

# step2: use inital model to crop vehicles
python tools/aicity20/weakly_supervised_crop_aug.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATALOADER.SAMPLER 'softmax' \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('./output/aicity20/0326-search/augmix/best.pth')"

# AIC20_ReID_cropped will be saved at './output/aicity20/0326-search/augmix/'
# dont forget to uncomment aicity20.py line 49 # train += self._process_dir(self.train_aug_dir, self.list_train_path, self.train_label_path, relabel=False)

````
- after all works have be done, data folder should look like
````
-Vehicle
--AIC20_ReID
--AIC20_ReID_Simulation
--AIC20_ReID_Cropped
````

## Download pretrain model
We use [ResNet-ibn](https://github.com/XingangPan/IBN-Net) as backbone.
Download ImageNet pretrain model at [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) 



## Train
- **Vehicle ReID.** Train three models respectively(resnet50, resnet101, resnext101),
```
bash ./scripts/aicity20/train.sh
```
- **Orientation ReID** Train orientation ReID model
```
bash ./scripts/aicity20/ReOriID.sh
```
- **Camera ReID** Train camera ReID model
```
bash ./scripts/aicity20/ReCamID.sh
```
you can either download our trained [models](https://drive.google.com/open?id=1W8nw3GEYyxZiuDSk_wdXTErHFxtfKfKI)


## Test and ensemble
- generate orientation and camera similarity matrix
```
bash ./scripts/aicity20/generate_matrix.sh
```
- generate vehicle distance matrix

```
bash ./scripts/aicity20/test.sh
```

- ensemble three distmat from three models
```
python ./tools/aicity20/multi_model_ensemble.py
```

