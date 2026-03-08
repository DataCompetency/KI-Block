import sys
import numpy.core
# Compatibility shim: .npz files saved with NumPy >= 2.0 reference numpy._core,
# which does not exist in NumPy 1.x. Map it to numpy.core instead.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray

from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GenomeShardDataset(Dataset):
    def __init__(self, shards_dir: str, split: str):
        self.paths = sorted(Path(shards_dir).glob(f"{split}_shard_*.npz"))
        if not self.paths:
            raise ValueError(f"No shard files found for split='{split}' in {shards_dir}")

        self._index: List[Tuple[int, int]] = []
        self._sizes: List[int] = []
        for shard_idx, path in enumerate(self.paths):
            with np.load(path, allow_pickle=True) as data:
                n = int(data["x"].shape[0])
            self._sizes.append(n)
            for local_idx in range(n):
                self._index.append((shard_idx, local_idx))

        self._cache_shard_idx = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int):
        if self._cache_shard_idx == shard_idx and self._cache_data is not None:
            return self._cache_data
        with np.load(self.paths[shard_idx], allow_pickle=True) as data:
            shard_data = {k: data[k] for k in data.files}
        self._cache_shard_idx = shard_idx
        self._cache_data = shard_data
        return shard_data

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._index[idx]
        data = self._load_shard(shard_idx)

        x = torch.from_numpy(data["x"][local_idx]).float()
        y = torch.from_numpy(data["y"][local_idx]).long()
        metadata = {
            "window_id": str(data["window_ids"][local_idx]),
            "chrom": str(data["chroms"][local_idx]),
            "start": int(data["starts"][local_idx]),
            "end": int(data["ends"][local_idx]),
        }
        return x, y, metadata


def make_batch_generator(
    shards_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = GenomeShardDataset(shards_dir=shards_dir, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

