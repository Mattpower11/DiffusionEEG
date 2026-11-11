import warnings
from typing import Dict, Tuple

import mne
import numpy as np

from src.conf import EEG_preprocessing_conf


class BCI2bEpochPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config

    def get_motor_imagery_events(
        self,
        raw_full: mne.io.BaseRaw,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extract MNE events and resolve left/right event descriptions."""
        events, event_id = mne.events_from_annotations(raw_full, verbose="ERROR")
        # MNE may preserve descriptions as strings; do not assume mapping values.
        desc_to_code = {str(k): int(v) for k, v in event_id.items()}

        def find_code(target: str) -> int:
            matches = [
                code for desc, code in desc_to_code.items()
                if desc.strip() == target
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected annotation description {target!r}; got "
                    f"{sorted(desc_to_code)}. Inspect raw.annotations.description "
                    "and the event mapping for this GDF."
                )
            return matches[0]

        return events, {
            "left_hand": find_code("769"),
            "right_hand": find_code("770"),
        }

    def extract_epochs(
        self,
        raw_eeg: mne.io.BaseRaw,
        raw_full: mne.io.BaseRaw,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Create cue-relative epochs for left/right imagery and reject artifacts."""
        events, codes = self.get_motor_imagery_events(raw_full)
        selected = events[np.isin(events[:, 2], list(codes.values()))]
        if len(selected) == 0:
            raise ValueError("No 769/770 motor-imagery events found.")

        # Exclude expert-marked rejected trials (1023), associating each marker
        # with the trial delimited by consecutive 768 start-of-trial events.
        desc_to_code = {
            str(k).strip(): int(v)
            for k, v in mne.events_from_annotations(raw_full, verbose="ERROR")[1].items()
        }
        rejected_code = desc_to_code.get("1023")
        trial_start_code = desc_to_code.get("768")
        if rejected_code is not None and trial_start_code is not None:
            starts = events[events[:, 2] == trial_start_code, 0]
            rejected_positions = events[events[:, 2] == rejected_code, 0]
            rejected_cues = []
            for cue in selected:
                previous = starts[starts <= cue[0]]
                if len(previous) == 0:
                    continue
                trial_start = previous[-1]
                following = starts[starts > trial_start]
                trial_end = following[0] if len(following) else np.inf
                if np.any(
                    (rejected_positions >= trial_start)
                    & (rejected_positions < trial_end)
                ):
                    rejected_cues.append(int(cue[0]))
            if rejected_cues:
                selected = selected[~np.isin(selected[:, 0], rejected_cues)]
                warnings.warn(
                    f"Excluded {len(rejected_cues)} expert-marked rejected trial(s)."
                )
        if len(selected) == 0:
            raise ValueError(
                "No valid 769/770 trials remain after rejected-trial exclusion."
            )

        epochs = mne.Epochs(
            raw_eeg,
            selected,
            event_id=codes,
            tmin=self.config.tmin,
            tmax=self.config.tmax,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            reject={
                "eeg": self.config.amplitude_threshold_uv * 1e-6
            },
            flat={
                "eeg": self.config.flat_threshold_uv * 1e-6
            },
            verbose="ERROR",
        )

        data = epochs.get_data(copy=True)  # Volts, shape [N, 3, T]
        p2p = np.ptp(data, axis=-1)
        keep = np.all(p2p <= self.config.peak_to_peak_threshold_uv * 1e-6, axis=1)
        data = data[keep]
        event_codes = epochs.events[keep, 2]
        labels = np.where(event_codes == codes["left_hand"], 0, 1).astype(np.int64)

        if data.size == 0:
            raise ValueError("All epochs were rejected; review thresholds and timing.")

        if (
            self.config.sequence_length is not None
            and data.shape[-1] != self.config.sequence_length
        ):
            data = np.stack([
                np.stack([
                    np.interp(
                        np.linspace(0, 1, self.config.sequence_length),
                        np.linspace(0, 1, trial.shape[-1]),
                        trial[ch],
                    )
                    for ch in range(trial.shape[0])
                ])
                for trial in data
            ])

        return data.astype(np.float32), labels, codes
