import re
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mne
import numpy as np

from src.conf import EEG_preprocessing_conf


class BCI2bGDFPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config

    @staticmethod
    def _normalise_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def load_gdf_file(self, file_path: Path) -> mne.io.BaseRaw:
        """Load one GDF file and validate the nominal sampling frequency."""
        raw = mne.io.read_raw_gdf(str(file_path), preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if abs(sfreq - self.config.sfreq_expected) > self.config.sfreq_tolerance:
            raise ValueError(
                f"{file_path.name}: expected {self.config.sfreq_expected} Hz, "
                f"got {sfreq} Hz"
            )
        if not np.isfinite(raw.get_data()).all():
            raise ValueError(f"{file_path.name}: raw data contain NaN or Inf")
        return raw

    def annotate_and_fill_nonfinite_gaps(
        self,
        raw: mne.io.BaseRaw,
        guard_seconds: float = 1.0,
    ) -> mne.io.BaseRaw:
        """Mark non-finite gaps BAD and interpolate only to permit filtering.

        The official 2b documentation notes 100-sample missing gaps between runs
        in some representations. Epochs near these gaps are rejected later.
        """
        out = raw.copy().load_data()
        data = out._data
        bad_samples = ~np.isfinite(data).all(axis=0)
        if not bad_samples.any():
            return out
        edges = np.diff(np.r_[False, bad_samples, False].astype(np.int8))
        starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
        sfreq = float(out.info["sfreq"])
        onsets, durations = [], []
        for a, b in zip(starts, stops):
            for row in data:
                finite = np.isfinite(row)
                known = np.flatnonzero(finite)
                if len(known) < 2:
                    raise ValueError("Too few finite samples to interpolate a GDF gap.")
                row[a:b] = np.interp(np.arange(a, b), known, row[known])
            onset = max(0.0, a / sfreq - guard_seconds)
            end = min(out.n_times / sfreq, b / sfreq + guard_seconds)
            onsets.append(onset)
            durations.append(max(0.0, end - onset))
        bad = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=["BAD_nonfinite_gap"] * len(onsets),
            orig_time=None,
        )
        out.set_annotations(out.annotations + bad)
        warnings.warn(
            f"Found {len(onsets)} non-finite gap(s); interpolated for filtering "
            "and marked expanded intervals BAD so affected epochs are rejected."
        )
        return out

    def select_eeg_channels(
        self,
        raw: mne.io.BaseRaw,
        eeg_names: Optional[Sequence[str]] = None,
    ) -> Tuple[mne.io.BaseRaw, List[str], List[str]]:
        """
        Select the three EEG channels by name, never by array position.

        Exact normalized-name matching is used. If labels contain suffixes such as
        'C3..' or 'EEG-C3', edit eeg_channel_names / this resolver after inspecting
        raw.ch_names. Ambiguous or missing channels raise an informative error.
        """
        if eeg_names is None:
            eeg_names = self.config.eeg_channel_names

        available = list(raw.ch_names)
        selected: List[str] = []
        for requested in eeg_names:
            key = self._normalise_name(requested)
            matches = [
                name
                for name in available
                if self._normalise_name(name) == key
                or self._normalise_name(name).endswith(key)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot uniquely resolve EEG channel {requested!r}. "
                    f"Available channels: {available}. Inspect labels and configure "
                    "EEGConfig.eeg_channel_names."
                )
            selected.append(matches[0])

        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError(f"Expected exactly 3 distinct EEG channels, got {selected}")

        eog_names = [
            name for name in available
            if re.search(self.config.eog_channel_pattern, name)
        ]
        if set(selected) & set(eog_names):
            raise ValueError("An EOG channel was selected as an EEG input.")

        eeg_raw = raw.copy().pick(selected)
        eeg_raw.set_channel_types(
            {name: "eeg" for name in selected},
            on_unit_change="ignore",
        )
        return eeg_raw, selected, eog_names
