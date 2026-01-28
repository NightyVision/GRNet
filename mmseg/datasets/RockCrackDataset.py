from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RockCrackDataset(CustomDataset):
    CLASSES = (
        'background','crack')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(RockCrackDataset, self).__init__(
            img_suffix='.png', # 图像的格式
            seg_map_suffix='.png', #标注mask图像的格式
            reduce_zero_label=False, 
            **kwargs)

