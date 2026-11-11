from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BCI2bDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        stats: Optional[Dict[str, object]] = None,
        unconditional: bool = False,
    ):
        if data.ndim != 3 or data.shape[1] != 3:
            raise ValueError(f"Expected [N, 3, T], got {data.shape}")
        if len(data) != len(labels):
            raise ValueError("Data and labels have different lengths.")

        if stats is not None:
            mode = stats.get("mode")
            if mode == "trial":
                mean = data.mean(axis=-1, keepdims=True)
                std = np.maximum(data.std(axis=-1, keepdims=True), float(stats.get("eps", 1e-6)))
                self.data = ((data - mean) / std).astype(np.float32)
            else:
                mean = np.asarray(stats["mean"], dtype=np.float32)
                std = np.asarray(stats["std"], dtype=np.float32)
                self.data = ((data - mean) / std).astype(np.float32)
        else:
            self.data = data.astype(np.float32)

        self.labels = labels.astype(np.int64)
        self.unconditional = unconditional
        if not np.isfinite(self.data).all():
            raise ValueError("Dataset contains NaN or Inf.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.data[index]).to(torch.float32)
        if self.unconditional:
            return x
        return x, torch.tensor(self.labels[index], dtype=torch.long)