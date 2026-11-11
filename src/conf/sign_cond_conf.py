from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class sign_cond_conf:
    channels: int = 3
    num_classes: int = 2
    sequence_length: int = 512

    max_iterations: int = 1_000_000
    batch_size: int = 32
    save_every: int = 10_000

    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    learning_rate: float = 1e-4
