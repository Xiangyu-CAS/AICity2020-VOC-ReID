# encoding: utf-8
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .aicity20 import AICity20
from .aicity20_sim import AICity20Sim
from .aicity20_trainval import AICity20Trainval
from .aicity20_ReOri import AICity20ReOri
from .aicity20_ReCam import AICity20ReCam
from .aicity20_ReColor import AICity20ReColor
from .aicity20_ReType import AICity20ReType
from .dataset_loader import ImageDataset
from .bases import BaseImageDataset, apply_id_bias

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc-reid': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'aicity20': AICity20,
    'aicity20-sim': AICity20Sim,
    'aicity20-trainval': AICity20Trainval,
    'aicity20-ReOri': AICity20ReOri,
    'aicity20-ReCam': AICity20ReCam,
    'aicity20-ReColor': AICity20ReColor,
    'aicity20-ReType': AICity20ReType,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
