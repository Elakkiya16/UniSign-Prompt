from pathlib import Path
from .continuous import ContinuousSignDataset

class RWTHPhoenixDataset(ContinuousSignDataset):
    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(Path(data_root)/f"rwth_{split}_meta.csv", data_root,
                         "DGS", True, True, 9, **kwargs)
