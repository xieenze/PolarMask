from .coco import CocoDataset
from .registry import DATASETS
from .coco_seg import Coco_Seg_Dataset


@DATASETS.register_module
class CityscapesDataset(Coco_Seg_Dataset):

    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')
