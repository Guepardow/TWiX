from .MOT17 import MOT17
from .MOT20 import MOT20
from .KITTIMOT import KITTIMOT
from .DanceTrack import DanceTrack

DATASETS = {
    'MOT17': MOT17,
    'MOT20': MOT20,
    'KITTIMOT': KITTIMOT,
    'DanceTrack': DanceTrack,
}


def init_scene(dataset, subset):
    """
    Initialize a subset of a dataset

    :param dataset: name of the tracking dataset
    :param subset: set (e.g. train, val, test)
    """

    if dataset not in DATASETS:
        raise KeyError(f"The dataset {dataset} is not known. Try one of the following: {DATASETS.keys()}")
    return DATASETS[dataset](subset)
