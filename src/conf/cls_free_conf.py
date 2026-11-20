from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class cls_free_conf:
    channels: int = 3
    num_classes: int = 2
    sequence_length: int = 500
    data_dir: str = "./../datasets/processed_BCICIV_2b_gdf"

    epochs: int = 100
    batch_size: int = 32

    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    learning_rate: float = 3e-4
