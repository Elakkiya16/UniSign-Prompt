from pathlib import Path
from .continuous import ContinuousSignDataset

class CSLDailyDataset(ContinuousSignDataset):
    def __init__(self, data_root, split="train", **kwargs):
        super().__init__(Path(data_root)/f"csl_daily_{split}_meta.csv", data_root,
                         "CSL", True, True, 10, **kwargs)
