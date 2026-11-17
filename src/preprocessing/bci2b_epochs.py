from typing import Dict, Tuple
import warnings

import mne
import numpy as np

from src.conf import EEG_preprocessing_conf


class BCI2bEpochPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config
        self.last_trial_report = []

    @staticmethod
    def _annotation_event_map(raw):
        _, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        return {str(k).strip(): int(v) for k, v in event_id.items()}

    @staticmethod
    def _find_code(desc_to_code, target):
        if target not in desc_to_code:
            raise ValueError(
                f"Expected annotation {target!r}; found {sorted(desc_to_code)}"
            )
        return desc_to_code[target]

    def get_motor_imagery_events(
        self, 
        raw_full: mne.io.BaseRaw
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extract MNE events and resolve left/right event descriptions."""
        events, event_id = mne.events_from_annotations(raw_full, verbose="ERROR")
        desc_to_code = {str(k).strip(): int(v) for k, v in event_id.items()}

        return events, {
            "left_hand": self._find_code(desc_to_code, "769"),
            "right_hand": self._find_code(desc_to_code, "770")
        }

    def extract_epochs(
        self,
        raw_eeg: mne.io.BaseRaw,
        raw_full: mne.io.BaseRaw
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Create cue-relative epochs for left/right imagery and reject artifacts."""
        
        self.last_trial_report = []
    
        if raw_eeg.info["sfreq"] != raw_full.info["sfreq"]:
            raise ValueError("raw_eeg and raw_full have different sampling frequencies.")
    
        if raw_eeg.first_samp != raw_full.first_samp:
            raise ValueError("raw_eeg and raw_full have different first samples.")

        if raw_eeg.n_times != raw_full.n_times:
            raise ValueError("raw_eeg and raw_full have different lengths.")

        if len(raw_eeg.ch_names) != 3:
            raise ValueError(
                f"Expected exactly 3 EEG channels, "
                f"found {len(raw_eeg.ch_names)}: {raw_eeg.ch_names}"
            )

        if not all(ch_type == "eeg" for ch_type in raw_eeg.get_channel_types()):
            raise ValueError(
                "raw_eeg must contain only EEG channels."
            )
        
        events, codes = self.get_motor_imagery_events(raw_full)
        selected = events[np.isin(events[:, 2], list(codes.values()))]

        if len(selected) == 0:
            raise ValueError("No 769/770 motor-imagery events found.")

        desc_map = self._annotation_event_map(raw_full)
        rejected_code = desc_map.get("1023")
        trial_start_code = desc_map.get("768")

        starts = (
            events[events[:, 2] == trial_start_code, 0]
            if trial_start_code is not None else np.array([], dtype=int)
        )

        rejected_positions = (
            events[events[:, 2] == rejected_code, 0]
            if rejected_code is not None else np.array([], dtype=int)
        )

        # Se il dataset non contiene affatto il marker 768,
        # interrompiamo l'elaborazione.
        if trial_start_code is None:
            raise ValueError(
                "Marker 768 (trial start) not found in the dataset."
            )

        expert_rejection_reasons = {}

        for cue in selected:
            cue_sample = int(cue[0])

            # Cerca il marker 768 immediatamente precedente.
            previous = starts[starts <= cue_sample]

            if len(previous) == 0:
                expert_rejection_reasons[cue_sample] = "missing_trial_start_768"
                continue

            trial_start = previous[-1]

            if len(previous) >= 2:
                previous_start = previous[-2]

                cues_between_starts = selected[
                    (selected[:, 0] > previous_start) &
                    (selected[:, 0] < trial_start)
                ]

                if len(cues_between_starts) == 0:
                    warnings.warn(
                        f"Possible trial without motor-imagery cue: "
                        f"start={previous_start}, next_start={trial_start}, "
                        f"current_cue={cue_sample}."
                    )
            
            # Verifica che non ci sia già un altro cue tra il marker 768
            # e quello corrente: in tal caso il marker potrebbe appartenere
            # al trial precedente.
            previous_cues = selected[
                (selected[:, 0] > trial_start) &
                (selected[:, 0] < cue_sample)
            ]

            if len(previous_cues) > 0:
                expert_rejection_reasons[cue_sample] = "missing_trial_start_768"
                continue

            # Individua la fine del trial.
            following = starts[starts > trial_start]
            trial_end = following[0] if len(following) else np.inf

            # Controlla se il trial è stato rifiutato dal dataset.
            if np.any(
                (rejected_positions >= trial_start) &
                (rejected_positions < trial_end)
            ):
                expert_rejection_reasons[cue_sample] = "expert_marker_1023"

        # Build all non-expert-rejected epochs without automatic rejection so
        # each threshold can be audited independently.
        candidate = selected[
            ~np.isin(
                selected[:, 0],
                list(expert_rejection_reasons.keys())
            )
        ]
        
        if len(candidate) == 0:
            raise ValueError(
                "No valid trial candidates remain after marker validation. "
                f"Rejection reasons: {expert_rejection_reasons}"
            )

        epochs_all = mne.Epochs(
            raw_eeg, candidate, 
            event_id=codes,
            on_missing="ignore", 
            tmin=self.config.tmin,
            tmax=self.config.tmax, 
            baseline=None, 
            preload=True,
            reject_by_annotation=False, 
            reject=None, 
            flat=None, 
            verbose="ERROR"
        )

        epochs_bad_checked = mne.Epochs(
            raw_eeg, 
            candidate, 
            event_id=codes,
            on_missing="ignore", 
            tmin=self.config.tmin,
            tmax=self.config.tmax, 
            baseline=None, 
            preload=True,
            reject_by_annotation=True, 
            reject=None, 
            flat=None, 
            verbose="ERROR"
        )

        annotation_good_samples = set(int(x) for x in epochs_bad_checked.events[:, 0])

        all_data = epochs_all.get_data(copy=True)
        sfreq = float(raw_eeg.info["sfreq"])

        actual_t = len(epochs_all.times)

        if actual_t != all_data.shape[-1]:
            raise RuntimeError(
                "Incoerenza tra la griglia temporale MNE "
                "e i dati estratti."
            )

        target_t = self.config.sequence_length

        if target_t is not None and actual_t != target_t:
            raise ValueError(
                f"Epoch length mismatch: MNE extracted {actual_t} samples, "
                f"but the model expects {target_t}. "
                f"Check tmin={self.config.tmin}, "
                f"tmax={self.config.tmax}, sfreq={sfreq}."
            )

        sample_to_data = {
            int(sample): all_data[i]
            for i, sample in enumerate(epochs_all.events[:, 0])
        }
        
        records = []

        # Include expert-rejected trials in the audit.
        for cue in selected:
            sample = int(cue[0])
            code = int(cue[2])
            label = 0 if code == codes["left_hand"] else 1
            reasons = []

            if sample in expert_rejection_reasons:
                rejection_reason = expert_rejection_reasons[sample]

                records.append({
                    "event_sample": sample,
                    "event_time_seconds": sample / sfreq,
                    "label": label,
                    "class": "left_hand" if label == 0 else "right_hand",
                    "expert_rejected_1023": rejection_reason == "expert_marker_1023",
                    "annotation_bad": None,
                    "max_abs_amplitude_uv": None,
                    "peak_to_peak_uv": {},
                    "flat_channels": [],
                    "kept": False,
                    "rejection_reasons": [rejection_reason]
                })
                
                continue

            trial = sample_to_data.get(sample)
            if trial is None:
                records.append({
                    "event_sample": sample, 
                    "event_time_seconds": sample / sfreq,
                    "label": label, "class": "left_hand" if label == 0 else "right_hand",
                    "expert_rejected_1023": False, 
                    "annotation_bad": False,
                    "max_abs_amplitude_uv": None, 
                    "peak_to_peak_uv": {},
                    "flat_channels": [], 
                    "kept": False,
                    "rejection_reasons": ["epoch_out_of_bounds_or_unavailable"]
                })
                continue

            if not np.isfinite(trial).all():
                reasons = ["non_finite_values"]

                if sample not in annotation_good_samples:
                    reasons.append("bad_annotation_overlap")

                records.append({
                    "event_sample": sample,
                    "event_time_seconds": sample / sfreq,
                    "label": label,
                    "class": "left_hand" if label == 0 else "right_hand",
                    "expert_rejected_1023": False,
                    "annotation_bad": sample not in annotation_good_samples,
                    "max_abs_amplitude_uv": None,
                    "peak_to_peak_uv": {},
                    "flat_channels": [],
                    "kept": False,
                    "rejection_reasons": reasons
                })
                continue
            
            max_abs_uv = np.max(np.abs(trial), axis=1) * 1e6
            p2p_uv = np.ptp(trial, axis=1) * 1e6
            ch_names = list(raw_eeg.ch_names)
            flat_channels = [
                ch_names[i] for i, value in enumerate(p2p_uv)
                if value < self.config.flat_threshold_uv
            ]
            
            if sample not in annotation_good_samples:
                reasons.append("bad_annotation_overlap")
         
            if flat_channels:
                reasons.append("flat_threshold")
            
            if np.any(p2p_uv > self.config.peak_to_peak_threshold_uv):
                reasons.append("peak_to_peak_threshold")

            records.append({
                "event_sample": sample,
                "event_time_seconds": sample / sfreq,
                "label": label,
                "class": "left_hand" if label == 0 else "right_hand",
                "expert_rejected_1023": False,
                "annotation_bad": sample not in annotation_good_samples,
                "max_abs_amplitude_uv": float(np.max(max_abs_uv)),
                "peak_to_peak_uv": {
                    ch_names[i]: float(p2p_uv[i]) for i in range(len(ch_names))
                },
                "flat_channels": flat_channels,
                "kept": len(reasons) == 0,
                "rejection_reasons": reasons,
            })

        self.last_trial_report = records
        keep_samples = {
            int(r["event_sample"]) for r in records if r["kept"]
        }
        kept_rows = [
            i for i, sample in enumerate(epochs_all.events[:, 0])
            if int(sample) in keep_samples
        ]

        if not kept_rows:
            raise ValueError("All epochs were rejected; inspect trial_audit.csv.")

        data = all_data[kept_rows]
        event_codes = epochs_all.events[kept_rows, 2]
        labels = np.where(event_codes == codes["left_hand"], 0, 1).astype(np.int64)

        counts = {}

        for record in records:
            for reason in record["rejection_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        
        rejected = sum(not r["kept"] for r in records)
        
        if rejected:
            warnings.warn(
                f"Rejected {rejected}/{len(records)} trials. Reasons: {counts}"
            )

        return data.astype(np.float32), labels, codes