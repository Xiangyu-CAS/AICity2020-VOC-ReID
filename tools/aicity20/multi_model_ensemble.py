import numpy as np
import sys
sys.path.append('.')
from lib.data.datasets.aicity20 import AICity20
from tools.aicity20.submit import write_result_with_track


if __name__ == '__main__':
    dataset = AICity20('/home/zxy/data/ReID/vehicle')
    distmat_path = ['./output/aicity20/0409-ensemble/r50-320-circle/distmat.npy',
                    './output/aicity20/0409-ensemble/next101-320-circle/distmat.npy',
                    './output/aicity20/0409-ensemble/r101-320-circle/distmat.npy',
                    ]
    #cam_distmat = np.load('./output/aicity20/0407-ReCamID/distmat_submit.npy')
    #ori_distmat = np.load('./output/aicity20/0409-ensemble/ReTypeID/distmat_submit.npy')
    distmat = []
    for path in distmat_path:
        distmat.append(np.load(path))
    distmat = sum(distmat) / len(distmat)
    #distmat = distmat - 0.1 * cam_distmat - 0.1 * ori_distmat

    indices = np.argsort(distmat, axis=1)
    write_result_with_track(indices, './output/aicity20/submit/', dataset.test_tracks)

