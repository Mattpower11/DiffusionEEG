"""
Orchestrate BCI Competition IV 2b preprocessing and per-trial auditing.

Requires the session-aware config/epoch modules described by the project API.
Training and evaluation data must be preprocessed in separate calls and report
folders. This module never loads MATLAB labels itself or normalizes samples.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mne
import numpy as np

from src.conf import EEG_preprocessing_conf
from src.preprocessing.bci2b_epochs import BCI2bEpochPreprocessor
from src.preprocessing.bci2b_gdf import BCI2bGDFPreprocessor
from src.preprocessing.bci2b_signal import BCI2bSignalPreprocessor


class BCI2bPreprocessor:
    AUDIT_FIELDS = [
        "file", "trial_id", "trial_index", "trial_start_sample", "cue_sample",
        "feedback_sample", "n_feedback_events", "protocol", "split",
        "label_source", "event_sample", "event_time_seconds", "label", "class",
        "expert_rejected_1023", "annotation_bad", "max_abs_amplitude_uv",
        "peak_to_peak_uv", "flat_channels", "kept", "rejection_reasons",
    ]

    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config
        self.gdf_preprocessor = BCI2bGDFPreprocessor(config)
        self.signal_preprocessor = BCI2bSignalPreprocessor(config)
        self.epoch_preprocessor = BCI2bEpochPreprocessor(config)
        self._file_reports = []
        self.unfiltered_epochs = None

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): BCI2bPreprocessor._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BCI2bPreprocessor._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return BCI2bPreprocessor._json_safe(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _require_session_spec(self, canonical_name: str):
        session_spec = getattr(self.config, "session_spec", None)
        if not callable(session_spec):
            raise RuntimeError(
                "EEG_preprocessing_conf.session_spec(name) is unavailable. "
                "Use the session-aware configuration module; do not infer session "
                "settings from the filename in bci2b_pipeline.py."
            )
        spec = session_spec(canonical_name)
        for name in (
            "subject_id", "session_id", "split", "protocol", "cue_annotations",
            "tmin", "tmax", "labels_source",
        ):
            if not hasattr(spec, name):
                raise RuntimeError(
                    f"session_spec({canonical_name!r}) is missing required attribute {name!r}."
                )
        return spec

    @staticmethod
    def _require_attribute(obj, attr, module):
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"{module} must provide {attr!r} in its session-aware/report-enabled "
                "implementation; bci2b_pipeline.py cannot generate this information."
            )
        return getattr(obj, attr)

    def _save_file_report(self, report):
        report_dir = Path(self.config.report_output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(report["file"]).stem
        
        path = report_dir / f"{stem}_report.json"
        path.write_text(
            json.dumps(self._json_safe(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        audit_path = report_dir / "trial_audit.csv"
        write_header = not audit_path.exists() or audit_path.stat().st_size == 0
        
        with audit_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.AUDIT_FIELDS)
            
            if write_header:
                writer.writeheader()
            
            for trial in report["trials"]:
                row = dict(trial)
                row["file"] = report["file"]
                
                for field in ("peak_to_peak_uv", "flat_channels", "rejection_reasons"):
                    row[field] = json.dumps(self._json_safe(row.get(field)))
                
                writer.writerow({key: row.get(key) for key in self.AUDIT_FIELDS})

    def preprocess_file(
        self,
        file_path: Path,
        collect_unfiltered: bool = False,
        *,
        labels_path: Optional[Union[str, Path]] = None,
        session_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        file_path = Path(file_path)
        canonical_name = session_name if session_name is not None else file_path.name
        spec = self._require_session_spec(canonical_name)
        self.unfiltered_epochs = None

        raw_full = self.gdf_preprocessor.load_gdf_file(file_path)
        raw_full = self.gdf_preprocessor.annotate_and_fill_nonfinite_gaps(raw_full)
        raw_eeg, eeg_names, eog_names = self.gdf_preprocessor.select_eeg_channels(raw_full)
        
        filtered = self.signal_preprocessor.filter_eeg_signal(raw_eeg)
        cleaned = self.signal_preprocessor.remove_artifacts(filtered, raw_full, eog_names)
        
        try:
            x, y, event_codes = self.epoch_preprocessor.extract_epochs(
                cleaned,
                raw_full,
                file_path=canonical_name,
                labels_path=labels_path,
                session_spec=spec,
            )
        except TypeError as exc:
            raise TypeError(
                f"{file_path.name}: bci2b_epochs.py must accept "
                "extract_epochs(raw_eeg, raw_full, *, file_path, labels_path, "
                "session_spec) in the multisession version."
            ) from exc

        trials = self._require_attribute(
            self.epoch_preprocessor, "last_trial_report", "bci2b_epochs.py"
        )
        label_source = self._require_attribute(
            self.epoch_preprocessor, "last_label_source", "bci2b_epochs.py"
        )
       
        # The supplied bci2b_signal.py does not expose last_filter_report or
        # last_artifact_report. Record the configured operation without inventing
        # ICA component IDs, scores, or claims of successful artifact removal.
        filter_report = getattr(self.signal_preprocessor, "last_filter_report", None)
       
        if filter_report is None:
            filter_report = {
                "source": "pipeline_config_and_filter_call",
                "l_freq_hz": self.config.l_freq,
                "h_freq_hz": self.config.h_freq,
                "notch_freq_hz": self.config.notch_freq,
                "method": "fir",
                "phase": "zero",
                "sampling_frequency_hz": float(filtered.info["sfreq"]),
                "note": "Configured filter parameters; module does not expose a diagnostic report.",
            }
        
        artifact_report = getattr(self.signal_preprocessor, "last_artifact_report", None)
        
        if artifact_report is None:
            mode = self.config.artifact_mode
            artifact_report = {
                "source": "pipeline_observation",
                "mode": mode,
                "correction_applied": False if mode == "basic" else None,
                "ica_success": None,
                "excluded_components": None,
                "eog_scores": None,
                "details_available": False,
                "note": (
                    "Basic mode leaves filtered continuous EEG unchanged; rejection occurs during epoching."
                    if mode == "basic" else
                    "The signal module does not expose ICA status/scores: correction cannot be verified from this report."
                ),
            }
        
        gap_report = self._require_attribute(
            self.gdf_preprocessor, "last_gap_report", "bci2b_gdf.py"
        )
        
        if not isinstance(trials, list) or not trials:
            raise ValueError(f"{file_path.name}: empty or invalid last_trial_report.")
        
        for index, trial in enumerate(trials):
            for field in ("event_sample", "label", "kept", "rejection_reasons"):
                if field not in trial:
                    raise ValueError(
                        f"{file_path.name}: trial {index} missing required audit field {field!r}."
                    )
        
        event_samples = [int(trial["event_sample"]) for trial in trials]
        
        if len(set(event_samples)) != len(event_samples):
            raise ValueError(f"{file_path.name}: duplicate event_sample values in trial audit.")
        
        kept_trials = [trial for trial in trials if bool(trial["kept"])]
        
        if len(x) != len(y) or len(y) != len(kept_trials):
            raise ValueError(
                f"{file_path.name}: mismatch X={len(x)}, y={len(y)}, "
                f"audit kept={len(kept_trials)}."
            )
        
        if np.asarray(x).ndim != 3 or x.shape[1:] != (3, 500):
            raise ValueError(f"{file_path.name}: expected [N, 3, 500], got {x.shape}.")
        
        if not np.isfinite(x).all():
            raise ValueError(f"{file_path.name}: X contains NaN or Inf.")
        
        if not np.all(np.isin(y, (0, 1))):
            raise ValueError(f"{file_path.name}: y must contain only labels 0 and 1.")
        
        # Audit order must match the model output, not just the count. The epoch
        # module owns trial identities and has to preserve original trial indices.
        audit_labels = np.asarray([int(trial["label"]) for trial in kept_trials], dtype=int)
        
        if not np.array_equal(audit_labels, y):
            raise ValueError(
                f"{file_path.name}: retained trial labels/order do not match y. "
                "Check original-trial indexing in bci2b_epochs.py."
            )

        if collect_unfiltered:
            events = np.array(
                [
                    [int(trial["event_sample"]), 0, int(trial["label"]) + 1]
                    for trial in kept_trials
                ],
                dtype=int,
            )
            
            # No independent trial rejection: use exclusively retained trial IDs.
            epochs_before = mne.Epochs(
                raw_eeg,
                events,
                event_id={"left_hand": 1, "right_hand": 2},
                on_missing="ignore",
                tmin=spec.tmin,
                tmax=spec.tmax,
                baseline=None,
                preload=True,
                reject_by_annotation=False,
                reject=None,
                flat=None,
                verbose="ERROR",
            )
            
            before = epochs_before.get_data(copy=True)
            
            if not np.array_equal(epochs_before.events[:, 0], events[:, 0]):
                raise ValueError(
                    f"{file_path.name}: unfiltered epoch sample indices differ from retained trials."
                )
           
            if not np.array_equal(epochs_before.events[:, 2] - 1, y):
                raise ValueError(
                    f"{file_path.name}: unfiltered epoch labels differ from y."
                )
            
            if before.shape != x.shape:
                raise ValueError(
                    f"{file_path.name}: unfiltered shape {before.shape} differs from X {x.shape}."
                )
            
            self.unfiltered_epochs = before.astype(np.float32)

        reasons = Counter(
            str(reason)
            for trial in trials
            for reason in trial["rejection_reasons"]
        )
        
        session = {
            "subject_id": spec.subject_id,
            "session_id": spec.session_id,
            "protocol": spec.protocol,
            "split": spec.split,
            "cue_annotations": list(spec.cue_annotations),
            "epoch_tmin_seconds": spec.tmin,
            "epoch_tmax_seconds": spec.tmax,
            "label_source": label_source,
        }
        
        report = {
            "file": file_path.name,
            "canonical_session_name": canonical_name,
            "session": session,
            "sampling_frequency_hz": float(raw_full.info["sfreq"]),
            "eeg_channels": eeg_names,
            "eog_channels": eog_names,
            "filter": filter_report,
            "artifact_processing": artifact_report,
            "gaps": gap_report,
            "event_codes": event_codes,
            "trial_summary": {
                "initial_cued_trials": len(trials),
                "initial_original_trials": len(trials),
                "kept_trials": len(kept_trials),
                "rejected_trials": len(trials) - len(kept_trials),
                "rejection_counts_nonexclusive": dict(reasons),
                "kept_left_hand": int(np.sum(y == 0)),
                "kept_right_hand": int(np.sum(y == 1)),
            },
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "finite": bool(np.isfinite(x).all()),
            "trials": trials,
        }
        
        self._save_file_report(report)
        self._file_reports.append(report)
        
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
            "report_file": str(
                Path(self.config.report_output_dir) / f"{file_path.stem}_report.json"
            ),
            "session_id": spec.session_id,
            "protocol": spec.protocol,
            "split": spec.split,
            "label_source": label_source,
            "canonical_session_name": canonical_name,
        }
        
        return x, y, metadata

    def preprocess_dataset(
        self,
        gdf_files: Sequence[Path],
        collect_unfiltered: bool = False,
        *,
        labels_path: Optional[Union[str, Path]] = None,
        session_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        files = [Path(p) for p in gdf_files]
        
        if not files:
            raise ValueError("No GDF files supplied.")
        
        if len({p.name for p in files}) != len(files):
            raise ValueError("Duplicate GDF basenames would overwrite individual reports.")
        
        session_names = session_names or {}
        resolved = []
        
        for file_path in files:
            canonical_name = session_names.get(
                str(file_path), session_names.get(file_path.name, file_path.name)
            )
            spec = self._require_session_spec(canonical_name)
            resolved.append((file_path, canonical_name, spec))
        
        if len({name for _, name, _ in resolved}) != len(resolved):
            raise ValueError("Duplicate canonical session names are not allowed.")
       
        splits = {spec.split for _, _, spec in resolved}
        
        if len(splits) != 1:
            raise ValueError(
                "Do not mix training and evaluation GDF files in one preprocess_dataset() "
                f"call (found splits: {sorted(str(s) for s in splits)})."
            )

        self._file_reports = []
        self.unfiltered_epochs = None
        xs, ys, metadata, unfiltered = [], [], [], []
        report_dir = Path(self.config.report_output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in ("trial_audit.csv", "preprocessing_report.json"):
            target = report_dir / filename
            if target.exists():
                target.unlink()

        for file_path, canonical_name, _ in resolved:
            x, y, meta = self.preprocess_file(
                file_path,
                collect_unfiltered=collect_unfiltered,
                labels_path=labels_path,
                session_name=canonical_name,
            )
            xs.append(x)
            ys.append(y)
            metadata.append(meta)
            
            if collect_unfiltered:
                unfiltered.append(self.unfiltered_epochs)
        
        lengths = {x.shape[-1] for x in xs}
       
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent epoch lengths: {sorted(lengths)}.")
        
        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        
        if len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError("Concatenated X/y mismatch or non-finite EEG data.")
        
        if X.ndim != 3 or X.shape[1:] != (3, 500):
            raise ValueError(f"Expected [N, 3, 500], got {X.shape}.")
        
        if collect_unfiltered:
            self.unfiltered_epochs = np.concatenate(unfiltered, axis=0).astype(np.float32)
            if self.unfiltered_epochs.shape != X.shape:
                raise ValueError("Concatenated unfiltered epochs do not match X shape.")
        
        counts = Counter()
        
        for report in self._file_reports:
            counts.update(report["trial_summary"]["rejection_counts_nonexclusive"])
        
        aggregate = {
            "split": next(iter(splits)),
            "n_files": len(self._file_reports),
            "initial_cued_trials": sum(
                r["trial_summary"]["initial_cued_trials"] for r in self._file_reports
            ),
            "initial_original_trials": sum(
                r["trial_summary"]["initial_original_trials"] for r in self._file_reports
            ),
            "kept_trials": sum(
                r["trial_summary"]["kept_trials"] for r in self._file_reports
            ),
            "rejected_trials": sum(
                r["trial_summary"]["rejected_trials"] for r in self._file_reports
            ),
            "rejection_counts_nonexclusive": dict(counts),
            "n_epochs_total": int(len(y)),
            "class_counts": {
                "left_hand": int(np.sum(y == 0)),
                "right_hand": int(np.sum(y == 1)),
            },
            "files": [
                {key: val for key, val in report.items() if key != "trials"}
                for report in self._file_reports
            ],
        }
        
        (report_dir / "preprocessing_report.json").write_text(
            json.dumps(self._json_safe(aggregate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        return X, y, metadata
