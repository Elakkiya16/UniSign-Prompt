from pathlib import Path
from .continuous import ContinuousSignDataset

class How2SignDataset(ContinuousSignDataset):
    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(Path(data_root)/f"how2sign_{split}_meta.csv", data_root,
                         "ASL", False, False, 0, **kwargs)
