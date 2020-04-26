### Prepare Data
- Download [CityFlow](https://www.aicitychallenge.org/) and [VehicleX](https://github.com/yorkeyao/VehicleX)
, rename them to 'AIC20_ReID' and 'AIC20_ReID_Simulation' respectively.
- Weakly Supervised crop Augmentation. Crop vehicle in image via weakly supervised method, 
a vehicle ReID pretrain model is needed to generate attention map. 
````
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

