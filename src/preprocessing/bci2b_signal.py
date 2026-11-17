import warnings
from typing import Sequence
from mne.preprocessing import ICA
from sklearn.exceptions import ConvergenceWarning

import mne
import numpy as np

from src.conf import EEG_preprocessing_conf


class BCI2bSignalPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config
        self.last_filter_report = {}
        self.last_artifact_report = {}

    def filter_eeg_signal(self, raw_eeg: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Filter continuous EEG and store the effective filter configuration."""
        filtered = raw_eeg.copy()
        if self.config.notch_freq is not None and self.config.notch_freq < filtered.info["sfreq"] / 2:
            filtered.notch_filter(
                freqs=[self.config.notch_freq], picks="eeg", method="fir",
                phase="zero", verbose="ERROR"
            )
        filtered.filter(
            l_freq=self.config.l_freq, h_freq=self.config.h_freq, picks="eeg",
            method="fir", phase="zero", fir_design="firwin", verbose="ERROR"
        )
        self.last_filter_report = {
            "l_freq_hz": self.config.l_freq,
            "h_freq_hz": self.config.h_freq,
            "notch_freq_hz": self.config.notch_freq,
            "method": "FIR",
            "phase": "zero",
            "sampling_frequency_hz": float(filtered.info["sfreq"]),
        }
        return filtered

    def remove_artifacts(
        self,
        raw_eeg: mne.io.BaseRaw,
        raw_full: mne.io.BaseRaw,
        eog_names: Sequence[str],
    ) -> mne.io.BaseRaw:
        """Apply configured artifact strategy and always retain an audit record."""
        report = {
            "mode": self.config.artifact_mode,
            "ica_status": "not_started",
            "correction_applied": False,
            "ica_attempted": False,
            "ica_success": None,

            "n_eeg_channels": len(raw_eeg.ch_names),
            "n_components": 0,
            "excluded_components": [],
            "detected_eog_components": [],

            "eog_channels": list(eog_names),
            "eog_scores": {},

            "eeg_rank": None,
            "ica_converged": None,
            "ica_iterations": None,
            "ica_max_iterations": None,
            "ica_convergence_warnings": [],

            "signal_change_rms": None,
            "signal_correlation": None,

            "failure_reason": None,
            "skip_reason": None,
            "notes": []
        }
        
        if self.config.artifact_mode == "basic":
            report["notes"].append(
                "Basic mode does not correct continuous EEG; epoch rejection is performed later."
            )
            self.last_artifact_report = report
            return raw_eeg
        
        if self.config.artifact_mode != "ica":
            raise ValueError("artifact_mode must be 'basic' or 'ica'")

        if len(eog_names) == 0:
            report["skip_reason"] = "No EOG channels found."
            report["ica_status"] = "skipped_no_eog"
            report["notes"].append(
                "ICA skipped because EOG channels are unavailable."
            )

            self.last_artifact_report = report

            if self.config.ica_strict:
                raise ValueError("No EOG channels found for ICA.")

            warnings.warn(report["skip_reason"])
            return raw_eeg

        try:
            data = raw_eeg.get_data(picks="eeg")

            if not np.isfinite(data).all():
                raise ValueError(
                    "EEG contains NaN or Inf before ICA"
                )

            report["ica_attempted"] = True
            report["ica_status"] = "running"

            work = raw_eeg.copy()
            
            fit_raw = work.copy().filter(
                l_freq=1.0, h_freq=None, picks="eeg", verbose="ERROR"
            )
            
            fit_data = fit_raw.get_data(picks="eeg")

            rank = np.linalg.matrix_rank(
                np.cov(fit_data)
            )

            report["eeg_rank"] = int(rank)

            if rank < 3:
                raise ValueError(
                    f"EEG rank insufficient for 3-component ICA: {rank}"
                )

            max_iter = 1000
            ica = ICA(
                n_components=min(3, len(work.ch_names)),
                method="fastica",
                random_state=self.config.random_seed,
                max_iter=max_iter,
            )
            
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                ica.fit(
                    fit_raw,
                    picks="eeg",
                    verbose="ERROR"
                )

            report["n_components"] = int(ica.n_components_)
            report["ica_iterations"] = int(ica.n_iter_)

            convergence_warnings = [
                str(w.message)
                for w in caught
                if issubclass(w.category, ConvergenceWarning)
            ]

            report["ica_max_iterations"] = max_iter
            report["ica_converged"] = (
                len(convergence_warnings) == 0
                and ica.n_iter_ < max_iter
            )

            report["ica_convergence_warnings"] = convergence_warnings

            if not report["ica_converged"]:
                raise RuntimeError(
                    "ICA did not converge within "
                    f"{max_iter} iterations. "
                    f"Actual iterations: {ica.n_iter_}"
                )

            eog_raw = raw_full.copy().pick(list(eog_names))

            if (eog_raw.info["sfreq"] == work.info["sfreq"] 
                and eog_raw.first_samp != work.first_samp):
                    raise ValueError(
                        "EEG and EOG have different initial sample indices: "
                        f"EEG={work.first_samp}, "
                        f"EOG={eog_raw.first_samp}"
                    )

            if abs(eog_raw.info["sfreq"] - work.info["sfreq"]) > 1e-6:
                eog_raw.resample(work.info["sfreq"], npad="auto", verbose="ERROR")
            
            eog_raw.set_channel_types(
                {name: "eog" for name in eog_raw.ch_names}, on_unit_change="ignore"
            )

            if eog_raw.n_times != work.n_times:
                raise ValueError(
                    "EEG/EOG sample count mismatch: "
                    f"EEG={work.n_times}, "
                    f"EOG={eog_raw.n_times}"
                )

            if not np.allclose(
                eog_raw.times,
                work.times,
                rtol=0,
                atol=1e-6
            ):
                raise ValueError(
                    "EEG and EOG have different time axes"
                )
            
            combined = work.copy().add_channels([eog_raw], force_update_info=True)
            
            eog_inds = set()

            for eog_name in eog_raw.ch_names:
                inds, scores = ica.find_bads_eog(
                    combined,
                    ch_name=eog_name,
                    measure="correlation",
                    threshold=self.config.ica_eog_threshold,
                    verbose="ERROR"
                )

                eog_inds.update(int(i) for i in inds)

                report["eog_scores"][eog_name] = (
                    np.asarray(scores)
                    .astype(float)
                    .tolist()
                )

            report["detected_eog_components"] = sorted(eog_inds)

            if (
                self.config.ica_auto_remove
                and len(eog_inds) > self.config.ica_max_components_to_remove
            ):
                raise ValueError(
                    "Too many ICA components marked for removal: "
                    f"{len(eog_inds)} out of {ica.n_components_}. "
                    "Automatic removal aborted."
                )

            if eog_inds and self.config.ica_auto_remove:

                ica.exclude = sorted(eog_inds)

                ica.apply(
                    work,
                    verbose="ERROR"
                )

                report["excluded_components"] = sorted(eog_inds)
                report["correction_applied"] = True

            elif eog_inds:
                report["notes"].append(
                    "Potential ocular components detected, "
                    "but automatic removal is disabled."
                )

            else:
                report["notes"].append("No ocular components passed the detection threshold.")

            original = raw_eeg.get_data(picks="eeg")
            corrected = work.get_data(picks="eeg")

            difference = corrected - original

            rms_values = (
                np.sqrt(np.mean(difference**2, axis=1)) * 1e6
            )

            report["signal_change_rms"] = {
                channel: float(value)
                for channel, value in zip(
                    raw_eeg.ch_names,
                    rms_values
                )
            }

            correlations = []

            for i in range(original.shape[0]):
                orig = original[i]
                corr_signal = corrected[i]

                if (np.std(orig) < 1e-12 or np.std(corr_signal) < 1e-12):
                    correlations.append(None)
                    continue

                corr = np.corrcoef(
                    orig,
                    corr_signal
                )[0, 1]

                correlations.append(float(corr) if np.isfinite(corr) else None)

            report["signal_correlation"] = {
                channel: value
                for channel, value in zip(
                    raw_eeg.ch_names,
                    correlations
                )
            }

            report["ica_success"] = True
            report["ica_status"] = "completed"

            self.last_artifact_report = report

            return work
        except Exception as exc:
            report["ica_success"] = False
            report["ica_status"] = "failed"
            report["correction_applied"] = False
            report["excluded_components"] = []
            report["signal_change_rms"] = None
            report["signal_correlation"] = None
            report["failure_reason"] = (f"{type(exc).__name__}: {exc}")
            report["notes"].append("ICA failed; filtered EEG returned unchanged.")

            self.last_artifact_report = report

            if self.config.ica_strict:
                raise

            warnings.warn(
                f"ICA failed; returning filtered EEG unchanged: {exc}"
            )

            return raw_eeg
