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
        self.last_gap_report = {}

    @staticmethod
    def _normalise_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def load_gdf_file(self, file_path: Path) -> mne.io.BaseRaw:
        """Load any training/evaluation GDF, retaining its original annotations.

        Non-finite samples are intentionally handled in the following step.
        """
        file_path = Path(file_path)
        raw = mne.io.read_raw_gdf(str(file_path), preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        
        if not np.isfinite(sfreq) or abs(sfreq - self.config.sfreq_expected) > self.config.sfreq_tolerance:
            raise ValueError(
                f"{file_path.name}: expected {self.config.sfreq_expected} Hz, "
                f"got {sfreq} Hz"
            )
        
        return raw

    @staticmethod
    def _interpolate_interval(row: np.ndarray, start: int, stop: int) -> None:
        """Linearly interpolate [start, stop) using two finite boundary samples."""
        if start <= 0 or stop >= len(row) or start >= stop:
            raise ValueError(
                f"Cannot interpolate interval [{start}, {stop}): "
                "finite anchors on both sides are required."
            )
        
        left, right = row[start - 1], row[stop]
        
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError(f"Non-finite interpolation anchors at [{start}, {stop}).")
        
        row[start:stop] = np.linspace(left, right, stop - start + 2)[1:-1]

    @staticmethod
    def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        return list(zip(np.flatnonzero(edges == 1).tolist(),
                        np.flatnonzero(edges == -1).tolist()))

    def annotate_and_fill_nonfinite_gaps(
        self,
        raw: mne.io.BaseRaw,
        guard_seconds: Optional[float] = None,
    ) -> mne.io.BaseRaw:
        """
        Interpolate verified acquisition gaps and annotate their guarded spans.

        A configured run-start event is only a *candidate*: the following
        configured number of samples must be finite and approximately constant on all recorded
        channels. The original event annotations are preserved. Added BAD_acq_skip
        annotations prevent default MNE filtering from crossing the gaps, and
        enable rejection of overlapping epochs with reject_by_annotation=True.

        This method is intended to be run once on each original GDF recording.
        The input Raw is never mutated.
        """
        self.last_gap_report = {}
        
        # A caller-supplied value takes precedence; otherwise use the shared
        # configuration (the existing pipeline explicitly passes 1.0 s).
        guard = float(
            self.config.gap_guard_seconds
            if guard_seconds is None
            else guard_seconds
        )
        
        if not np.isfinite(guard) or guard < 0:
            raise ValueError("guard_seconds must be finite and non-negative")

        n_gap = getattr(self.config, "expected_run_gap_samples", 100)
        
        if isinstance(n_gap, (bool, np.bool_)) or not isinstance(n_gap, (int, np.integer)) or n_gap <= 0:
            raise ValueError("expected_run_gap_samples must be a positive integer")
        
        n_gap = int(n_gap)

        tol = float(getattr(self.config, "sentinel_atol_volts", 1e-9))
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("sentinel_atol_volts must be finite and positive")

        out = raw.copy().load_data()
        marker_names = {"bad_acq_skip_run_gap", "bad_acq_skip_nonfinite_gap"}
        if any(str(desc).lower() in marker_names for desc in out.annotations.description):
            raise ValueError("Gaps already annotated by this preprocessor; reload original GDF")

        data = out._data
        sfreq = float(out.info["sfreq"])
        intervals: List[Tuple[int, int, str]] = []
        records = []

        original_bad = ~np.isfinite(data)
        n_nonfinite = int(original_bad.sum())
        
        for start, stop in self._runs(original_bad.any(axis=0)):
            intervals.append((start, stop, "nonfinite"))
            records.append({
                "kind": "nonfinite", "start_sample": start, "stop_sample": stop,
                "n_samples": stop - start, "duration_seconds": (stop - start) / sfreq,
                "operation": "interpolate_nonfinite_values_and_annotate_BAD",
                "interpolated": True,
            })

        # Get event positions through MNE, without changing annotation order or types.
        events, event_id = mne.events_from_annotations(out, verbose="ERROR")
        marker_description = str(getattr(self.config, "run_start_annotation", "32766")).strip()
        
        if not marker_description:
            raise ValueError("run_start_annotation must not be empty")
     
        marker_codes = [code for desc, code in event_id.items()
                        if str(desc).strip() == marker_description]
        candidates = 0
        rejected = []
        
        for code in marker_codes:
            samples = events[events[:, 2] == code, 0] - out.first_samp
            for sample in samples:
                candidates += 1
                start = int(sample)
                stop = start + n_gap
                reason = None
                ranges = None
                
                if start <= 0 or stop >= out.n_times:
                    reason = "out_of_bounds_or_no_interpolation_anchors"
                else:
                    segment = data[:, start:stop]
                    
                    if not np.isfinite(segment).all():
                        reason = "nonfinite_values"
                    else:
                        ranges = np.ptp(segment, axis=1)
                        
                        if not bool(np.all(ranges <= tol)):
                            reason = "not_constant"
                        elif any(start < b and stop > a for a, b, _ in intervals):
                            reason = "overlapping_interval"
                
                if reason is not None:
                    item = {"start_sample": start, "reason": reason}
                    
                    if ranges is not None:
                        item["channel_names"] = list(out.ch_names)
                        item["channel_ranges"] = ranges.tolist()
                    rejected.append(item)
                    continue

                intervals.append((start, stop, "run_gap"))
                records.append({
                    "kind": "run_gap_32766", "start_sample": start,
                    "stop_sample": stop, "n_samples": n_gap,
                    "duration_seconds": n_gap / sfreq,
                    "operation": "interpolate_all_channels_and_annotate_BAD",
                    "interpolated": True,
                    "channel_names": list(out.ch_names),
                    "channel_ranges": ranges.tolist(),
                })

        # For non-finite data, interpolate only the non-finite values per channel.
        # For verified run gaps, interpolate the entire constant block.
        for start, stop, kind in intervals:
            for channel_idx, row in enumerate(data):
                if kind == "nonfinite":
                    channel_runs = [(start + a, start + b)
                                    for a, b in self._runs(~np.isfinite(row[start:stop]))]
                else:
                    channel_runs = [(start, stop)]
                
                for a, b in channel_runs:
                    try:
                        self._interpolate_interval(row, a, b)
                    except ValueError as exc:
                        raise ValueError(
                            f"Interpolation failed for {out.ch_names[channel_idx]!r} "
                            f"at samples [{a}, {b}): {exc}"
                        ) from exc

        remaining = int((~np.isfinite(data)).sum())
        
        if remaining:
            raise RuntimeError(f"{remaining} NaN/Inf values remain after interpolation")

        guarded_mask = np.zeros(out.n_times, dtype=bool)
        guard_samples = int(round(guard * sfreq))
        
        for start, stop, kind in intervals:
            first = max(0, start - guard_samples)
            last = min(out.n_times, stop + guard_samples)
            guarded_mask[first:last] = True
            out.annotations.append(
                onset=out.first_time + first / sfreq,
                duration=(last - first) / sfreq,
                description=("BAD_acq_skip_run_gap" if kind == "run_gap"
                             else "BAD_acq_skip_nonfinite_gap"),
            )

        self.last_gap_report = {
            "n_gaps": len(records),
            "n_nonfinite_gaps": sum(r["kind"] == "nonfinite" for r in records),
            "n_run_gaps": sum(r["kind"] == "run_gap_32766" for r in records),
            "run_marker_description": marker_description,
            "n_run_marker_candidates": candidates,
            "n_run_marker_rejected": len(rejected),
            "rejected_run_markers": rejected,
            "n_affected_time_samples": sum(r["n_samples"] for r in records),
            "gap_duration_seconds": sum(r["n_samples"] for r in records) / sfreq,
            "n_remaining_nonfinite_values": remaining,
            "n_interpolated_values": n_nonfinite + sum(
                data.shape[0] * (stop - start)
                for start, stop, kind in intervals if kind == "run_gap"
            ),
            "n_bad_annotated_samples": int(guarded_mask.sum()),
            "guard_seconds": guard,
            "gaps": records,
        }
        
        if not marker_codes:
            warnings.warn(
                f"No '{marker_description}' annotations found: no run separators were checked.",
                stacklevel=2,
            )
        
        if records:
            warnings.warn(
                f"{len(records)} gap(s) interpolated and marked BAD.",
                stacklevel=2,
            )
       
        return out

    def select_eeg_channels(
        self,
        raw: mne.io.BaseRaw,
        eeg_names: Optional[Sequence[str]] = None,
    ) -> Tuple[mne.io.BaseRaw, List[str], List[str]]:
        """Return three named EEG signals, their names and auxiliary EOG names."""
        
        if eeg_names is None:
            eeg_names = self.config.eeg_channel_names
        
        available = list(raw.ch_names)
        selected: List[str] = []
        
        for requested in eeg_names:
            key = self._normalise_name(requested)
            
            if not key:
                raise ValueError(f"Invalid EEG channel name {requested!r}")
            
            exact = [n for n in available if self._normalise_name(n) == key]
            matches = exact if exact else [
                n for n in available if self._normalise_name(n).endswith(key)
            ]
            
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot uniquely resolve EEG channel {requested!r}; "
                    f"available: {available}"
                )
            
            selected.append(matches[0])
        
        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError(f"Expected exactly three distinct EEG channels, got {selected}")
        
        eog_names = [n for n in available if re.search(self.config.eog_channel_pattern, n)]
        
        if set(selected) & set(eog_names):
            raise ValueError("An EOG channel was selected as an EEG input")
        
        eeg_raw = raw.copy().pick(selected)
        eeg_raw.set_channel_types(
            {name: "eeg" for name in selected}, on_unit_change="ignore"
        )
        
        return eeg_raw, selected, eog_names
