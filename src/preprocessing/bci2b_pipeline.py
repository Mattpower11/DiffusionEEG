from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from src.conf import EEG_preprocessing_conf
from src.preprocessing.bci2b_epochs import BCI2bEpochPreprocessor
from src.preprocessing.bci2b_gdf import BCI2bGDFPreprocessor
from src.preprocessing.bci2b_signal import BCI2bSignalPreprocessor


class BCI2bPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.gdf_preprocessor = BCI2bGDFPreprocessor(config)
        self.signal_preprocessor = BCI2bSignalPreprocessor(config)
        self.epoch_preprocessor = BCI2bEpochPreprocessor(config)

    def preprocess_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        raw_full = self.gdf_preprocessor.load_gdf_file(file_path)
        raw_full = self.gdf_preprocessor.annotate_and_fill_nonfinite_gaps(
            raw_full,
            guard_seconds=1.0,
        )
        raw_eeg, eeg_names, eog_names = self.gdf_preprocessor.select_eeg_channels(raw_full)
        filtered = self.signal_preprocessor.filter_eeg_signal(raw_eeg)
        cleaned = self.signal_preprocessor.remove_artifacts(
            filtered,
            raw_full,
            eog_names,
        )
        x, y, event_codes = self.epoch_preprocessor.extract_epochs(cleaned, raw_full)
        metadata = {
            "file": file_path.name,
            "eeg_channels": eeg_names,
            "eog_channels": eog_names,
            "sfreq": float(raw_full.info["sfreq"]),
            "event_codes": event_codes,
            "n_epochs": int(len(y)),
            "class_counts": {
                "left_hand": int(np.sum(y == 0)),
                "right_hand": int(np.sum(y == 1)),
            },
            "shape": list(x.shape),
        }
        return x, y, metadata

    def preprocess_dataset(
        self,
        gdf_files: Sequence[Path],
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        xs, ys, metadata = [], [], []
        for file_path in gdf_files:
            x, y, meta = self.preprocess_file(Path(file_path))
            xs.append(x)
            ys.append(y)
            metadata.append(meta)
        if not xs:
            raise ValueError("No GDF files supplied.")
        lengths = {x.shape[-1] for x in xs}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent sequence lengths across files: {lengths}")
        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        if not np.isfinite(X).all():
            raise ValueError("Final data contain NaN or Inf.")
        return X, y, metadata
