from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

@dataclass
class EEG_preprocessing_conf:
    sfreq_expected: float = 250.0
    sfreq_tolerance: float = 0.01

    # Configure these after inspecting raw.ch_names. Exact names are preferred.
    # BCI IV 2b commonly uses bipolar derivations C3, Cz, C4, but the actual
    # GDF channel labels must be inspected; no positional assumption is made.
    eeg_channel_names: Tuple[str, ...] = ("C3", "Cz", "C4")
    eog_channel_pattern: str = r"(?i)\bEOG\b|EOG"

    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freq: Optional[float] = None

    # Screening sessions 01T/02T: cue is shown for 1.25 s, then the 4 s
    # imagery period starts. This default extracts 2 s inside that imagery
    # period relative to the 769/770 cue-onset event. Do not mix 03T feedback
    # sessions without defining a separate, protocol-appropriate epoch window.
    tmin: float = 1.25
    tmax: float = 3.246  # 500 samples at 250 Hz, inclusive endpoints

    sequence_length: Optional[int] = 500  # None preserves native epoch samples
    resample_sfreq: Optional[float] = None  # None preserves 250 Hz
    overlap: float = 0.0  # reserved for optional sliding-window extension

    artifact_mode: str = "basic"  # "basic" or "ica"
    amplitude_threshold_uv: float = 150.0
    peak_to_peak_threshold_uv: float = 200.0
    flat_threshold_uv: float = 0.5

    normalization: str = "train"  # "train" or "trial"
    eps: float = 1e-6

    random_seed: int = 42

