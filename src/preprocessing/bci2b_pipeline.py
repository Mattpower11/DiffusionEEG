import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from collections import Counter

import numpy as np

from src.conf import EEG_preprocessing_conf
from src.preprocessing.bci2b_epochs import BCI2bEpochPreprocessor
from src.preprocessing.bci2b_gdf import BCI2bGDFPreprocessor
from src.preprocessing.bci2b_signal import BCI2bSignalPreprocessor


class BCI2bPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config
        self.gdf_preprocessor = BCI2bGDFPreprocessor(config)
        self.signal_preprocessor = BCI2bSignalPreprocessor(config)
        self.epoch_preprocessor = BCI2bEpochPreprocessor(config)
        self._file_reports = []

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): BCI2bPreprocessor._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [BCI2bPreprocessor._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _save_file_report(self, report):
        report_dir = Path(self.config.report_output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(report["file"]).stem
        
        path = report_dir / f"{stem}_report.json"
        path.write_text(
            json.dumps(self._json_safe(report), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        audit_path = report_dir / "trial_audit.csv"
        fields = [
            "file", "event_sample", "event_time_seconds", "label", "class",
            "expert_rejected_1023", "annotation_bad", "max_abs_amplitude_uv",
            "peak_to_peak_uv", "flat_channels", "kept", "rejection_reasons"
        ]

        write_header = not audit_path.exists()
        with audit_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for record in report["trials"]:
                row = dict(record)
                row["file"] = report["file"]
                row["peak_to_peak_uv"] = json.dumps(row.get("peak_to_peak_uv", {}))
                row["flat_channels"] = json.dumps(row.get("flat_channels", []))
                row["rejection_reasons"] = json.dumps(row.get("rejection_reasons", []))
                writer.writerow({key: row.get(key) for key in fields})

    def preprocess_file(
            self, 
            file_path: Path
        ) -> Tuple[np.ndarray, np.ndarray, dict]:
        file_path = Path(file_path)
        raw_full = self.gdf_preprocessor.load_gdf_file(file_path)
        raw_full = self.gdf_preprocessor.annotate_and_fill_nonfinite_gaps(raw_full)
        raw_eeg, eeg_names, eog_names = self.gdf_preprocessor.select_eeg_channels(raw_full)
        
        filtered = self.signal_preprocessor.filter_eeg_signal(raw_eeg)
        cleaned = self.signal_preprocessor.remove_artifacts(filtered, raw_full, eog_names)
        
        x, y, event_codes = self.epoch_preprocessor.extract_epochs(cleaned, raw_full)

        ''' INIZIO TEST '''
        required = (
            (self.epoch_preprocessor, "last_trial_report"),
            (self.signal_preprocessor, "last_filter_report"),
            (self.signal_preprocessor, "last_artifact_report"),
            (self.gdf_preprocessor, "last_gap_report"),
        )

        for module, attribute in required:
            if not hasattr(module, attribute):
                raise RuntimeError(
                    f"{type(module).__name__} does not expose {attribute}. "
                    "Update all preprocessing modules to the report-enabled versions."
                )
        ''' FINE TEST '''

        trials = self.epoch_preprocessor.last_trial_report


        if not isinstance(trials, list) or not trials:
            raise ValueError(f"{file_path.name}: trial audit is empty or invalid")
        
        if len(trials) != len({int(t["event_sample"]) for t in trials}):
            raise ValueError(f"{file_path.name}: duplicate event samples in trial audit")
        
        n_kept = sum(bool(t["kept"]) for t in trials)
        
        if n_kept != len(y) or len(x) != len(y):
            raise ValueError(
                f"{file_path.name}: trial audit/data mismatch "
                f"(audit kept={n_kept}, X={len(x)}, y={len(y)})"
            )

        if not np.isfinite(x).all():
            raise ValueError(f"{file_path.name}: nonfinite values in epochs")

        rejection_counts = {}
        
        for trial in trials:
            for reason in trial["rejection_reasons"]:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        report = {
            "file": file_path.name,
            "sampling_frequency_hz": float(raw_full.info["sfreq"]),
            "eeg_channels": eeg_names,
            "eog_channels": eog_names,
            "filter": self.signal_preprocessor.last_filter_report,
            "artifact_processing": self.signal_preprocessor.last_artifact_report,
            "gaps": self.gdf_preprocessor.last_gap_report,
            "event_codes": event_codes,
            "trial_summary": {
                "initial_cued_trials": len(trials),
                "kept_trials": int(sum(t["kept"] for t in trials)),
                "rejected_trials": int(sum(not t["kept"] for t in trials)),
                "rejection_counts_nonexclusive": rejection_counts,
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
            "report_file": str(Path(self.config.report_output_dir) / f"{file_path.stem}_report.json"),
        }
        return x, y, metadata

    def preprocess_dataset(
        self, 
        gdf_files: Sequence[Path]
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        xs, ys, metadata = [], [], []
        self._file_reports = []

        gdf_files = [Path(p) for p in gdf_files]

        if not gdf_files:
            raise ValueError("No GDF files supplied.")
        
        if len({p.name for p in gdf_files}) != len(gdf_files):
            raise ValueError("Duplicate GDF basenames would overwrite individual reports.")
        
        report_dir = Path(self.config.report_output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Each dataset run starts a fresh audit, avoiding stale/duplicate rows.
        audit_path = report_dir / "trial_audit.csv"
        if audit_path.exists():
            audit_path.unlink()
        
        aggregate_path = report_dir / "preprocessing_report.json"
        if aggregate_path.exists():
            aggregate_path.unlink()
        
        for file_path in gdf_files:
            x, y, meta = self.preprocess_file(Path(file_path))
            xs.append(x)
            ys.append(y)
            metadata.append(meta)
        
        lengths = {x.shape[-1] for x in xs}
       
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent sequence lengths across files: {lengths}")
        
        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        
        if not np.isfinite(X).all():
            raise ValueError("Final data contain NaN or Inf.")

        report_dir = Path(self.config.report_output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        aggregate = {
            "n_files": len(self._file_reports),
            "initial_cued_trials": sum(r["trial_summary"]["initial_cued_trials"] for r in self._file_reports),
            "kept_trials": sum(r["trial_summary"]["kept_trials"] for r in self._file_reports),
            "rejected_trials": sum(r["trial_summary"]["rejected_trials"] for r in self._file_reports),
            "rejection_counts_nonexclusive": dict(sum(
                (Counter(r["trial_summary"]["rejection_counts_nonexclusive"]) for r in self._file_reports),
                Counter()
            )),
            "n_epochs_total": int(len(y)),
            "class_counts": {
                "left_hand": int(np.sum(y == 0)),
                "right_hand": int(np.sum(y == 1)),
            },
            "files": [
                {k: v for k, v in report.items() if k != "trials"}
                for report in self._file_reports
            ],
        }
        
        aggregate_path.write_text(
            json.dumps(self._json_safe(aggregate), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        return X, y, metadata
