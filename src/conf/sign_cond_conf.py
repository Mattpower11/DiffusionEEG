from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class sign_cond_conf:
    channels: int = 3
    num_classes: int = 2
    sequence_length: int = 500
    data_dir: str = "./../datasets/processed_BCICIV_2b_gdf"

    max_iterations: int = 1_000_000
    batch_size: int = 32
    save_every: int = 10_000

    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    learning_rate: float = 1e-4
    snr_db: int = 20

    test_samples_per_class: int = 16
    test_batch_size: int = 4
    test_sampling_timesteps: int = 2000
    test_ds_epochs: int = 30
    test_seed: int = 42
    test_results_dir: str = "./results/eeg_test"
    test_snr_db: float = 20.0
