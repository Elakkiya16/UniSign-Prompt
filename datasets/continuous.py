"""Load video, feature and token data from explicit metadata paths."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

LANGUAGES = {'ASL': 0, 'DGS': 1, 'ISL': 2, 'CSL': 3}

class ContinuousSignDataset(Dataset):
    def __init__(self, meta_path, data_root, language, use_gloss, use_signers,
                 num_signers=0, fps=25, resolution=224, bos_id=1, eos_id=2,
                 pad_id=0, text_vocab_size=None, gloss_vocab_size=None):
        self.root = Path(data_root)
        self.language, self.use_gloss, self.use_signers = LANGUAGES[language], use_gloss, use_signers
        self.num_signers, self.fps, self.resolution = num_signers, fps, resolution
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.text_vocab_size, self.gloss_vocab_size = text_vocab_size, gloss_vocab_size
        with open(meta_path, newline='', encoding='utf-8') as stream:
            self.samples = list(csv.DictReader(stream))
        if not self.samples:
            raise ValueError(f"No samples in {meta_path}")
        for row in self.samples:
            required = ['sample_id', 'text_path', 'gloss_path'] if use_gloss else ['sample_id', 'text_path']
            if any(not row.get(key) for key in required) or not (row.get('video_path') or row.get('feature_path')):
                raise ValueError("Metadata requires sample_id, text_path, video_path or feature_path, and gold gloss_path when enabled")
            if use_signers and (not row.get('signer_id') or not 0 <= int(row['signer_id']) < num_signers):
                raise ValueError("Signer-supervised training requires valid, consistently mapped signer IDs")

    def __len__(self):
        return len(self.samples)

    def _tokens(self, path, vocabulary):
        array = np.load(self.root/path, allow_pickle=False)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Token arrays must be one-dimensional integers")
        tokens = torch.as_tensor(array.astype(np.int64))
        if len(tokens) < 2 or tokens[0] != self.bos_id or tokens[-1] != self.eos_id or (tokens == self.pad_id).any():
            raise ValueError("Unpadded token arrays must start with BOS and end with EOS")
        if (tokens < 0).any() or (vocabulary is not None and (tokens >= vocabulary).any()):
            raise ValueError("Token IDs exceed the configured vocabulary")
        return tokens

    def _video(self, path):
        import cv2
        capture = cv2.VideoCapture(str(self.root/path))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or source_fps <= 0:
            capture.release()
            raise ValueError(f"Cannot read video/FPS: {self.root/path}")
        frames, index, next_time = [], 0, 0.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                time = index/source_fps
                if time + 1e-8 >= next_time:
                    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.resolution, self.resolution))
                    frames.append(torch.from_numpy(frame.copy()).permute(2,0,1).float()/127.5-1)
                    next_time += 1/self.fps
                index += 1
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"Empty video: {path}")
        return torch.stack(frames)

    def __getitem__(self, index):
        row = self.samples[index]
        visual = torch.as_tensor(np.load(self.root/row['feature_path'], allow_pickle=False), dtype=torch.float32) if row.get('feature_path') else self._video(row['video_path'])
        if visual.ndim not in (2,4) or visual.size(0) == 0 or not torch.isfinite(visual).all():
            raise ValueError("Visual inputs must be nonempty finite features or RGB frames")
        return {'sample_id': row['sample_id'], 'visual': visual,
                'language_ids': self.language,
                'signer_ids': int(row['signer_id']) if self.use_signers else None,
                'text': self._tokens(row['text_path'], self.text_vocab_size),
                'gloss': self._tokens(row['gloss_path'], self.gloss_vocab_size) if self.use_gloss else None}

def collate_samples(samples, pad_id=0):
    lengths = torch.tensor([len(s['visual']) for s in samples])
    result = {'sample_id': [s['sample_id'] for s in samples],
              'visual': pad_sequence([s['visual'] for s in samples], batch_first=True),
              'frame_mask': torch.arange(int(lengths.max()))[None] >= lengths[:, None],
              'language_ids': torch.tensor([s['language_ids'] for s in samples])}
    for key in ('text', 'gloss'):
        result[key] = pad_sequence([s[key] for s in samples], batch_first=True, padding_value=pad_id) if samples[0][key] is not None else None
    signer = [s['signer_ids'] for s in samples]
    if any(s is None for s in signer) and not all(s is None for s in signer):
        raise ValueError("Use uniformly annotated batches; do not fabricate missing signer labels")
    result['signer_ids'] = None if signer[0] is None else torch.tensor(signer)
    return result
