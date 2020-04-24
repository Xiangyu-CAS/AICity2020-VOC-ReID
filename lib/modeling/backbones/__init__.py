from .resnet import resnet50
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, se_resnet101_ibn_a
from .resnext_ibn_a import resnext50_ibn_a, resnext101_ibn_a
from .resnest import resnest50
from .regnet.regnet import regnety_800mf, regnety_1600mf, regnety_3200mf

factory = {
    'resnet50': resnet50,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnext101_ibn_a': resnext101_ibn_a,
    'resnest50': resnest50,
    'regnety_800mf': regnety_800mf,
    'regnety_1600mf': regnety_1600mf,
    'regnety_3200mf': regnety_3200mf,
}
def build_backbone(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)