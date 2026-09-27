from pathlib import Path
from .continuous import ContinuousSignDataset

class ISLCSLTRDataset(ContinuousSignDataset):
    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(Path(data_root)/f"isl_{split}_meta.csv", data_root,
                         "ISL", True, True, 7, **kwargs)
