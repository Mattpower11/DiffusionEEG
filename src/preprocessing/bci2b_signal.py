import warnings
from typing import Sequence

import mne

from src.conf import EEG_preprocessing_conf


class BCI2bSignalPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config

    def filter_eeg_signal(self, raw_eeg: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Filter continuous EEG; preserve native sampling rate."""
        filtered = raw_eeg.copy()
        if (
            self.config.notch_freq is not None
            and self.config.notch_freq < filtered.info["sfreq"] / 2
        ):
            filtered.notch_filter(
                freqs=[self.config.notch_freq],
                picks="eeg",
                method="fir",
                phase="zero",
                verbose="ERROR",
            )
        filtered.filter(
            l_freq=self.config.l_freq,
            h_freq=self.config.h_freq,
            picks="eeg",
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
        return filtered

    def remove_artifacts(
        self,
        raw_eeg: mne.io.BaseRaw,
        raw_full: mne.io.BaseRaw,
        eog_names: Sequence[str],
    ) -> mne.io.BaseRaw:
        """
        Basic mode: leave continuous data unchanged; epoch rejection is performed
        later using amplitude/flatness thresholds.

        ICA mode: intentionally conservative. With only 3 EEG channels ICA has
        very limited rank and is not a reliable general-purpose artifact separator.
        EOG channels are used as reference signals for component scoring. The
        function excludes only components strongly correlated with EOG; inspect
        before using for scientific conclusions.
        """
        if self.config.artifact_mode == "basic":
            return raw_eeg
        if self.config.artifact_mode != "ica":
            raise ValueError("artifact_mode must be 'basic' or 'ica'")

        if len(eog_names) == 0:
            warnings.warn("No EOG channels found; ICA ocular scoring unavailable.")
            return raw_eeg
        if len(raw_eeg.ch_names) < 3:
            warnings.warn("Too few EEG channels for ICA; skipping.")
            return raw_eeg

        # ICA on 3 channels is low-dimensional. Do not pretend it can clean all
        # muscle/electrical artifacts. EOG scoring is only an optional heuristic.
        from mne.preprocessing import ICA

        work = raw_eeg.copy()
        # Fit on high-pass copy as recommended for ICA stability; apply solution
        # to the filtered EEG. This is still exploratory with only 3 channels.
        fit_raw = work.copy().filter(
            l_freq=1.0,
            h_freq=None,
            picks="eeg",
            verbose="ERROR",
        )
        ica = ICA(
            n_components=3,
            method="fastica",
            random_state=self.config.random_seed,
            max_iter="auto",
        )
        ica.fit(fit_raw, picks="eeg", verbose="ERROR")

        eog_raw = raw_full.copy().pick(list(eog_names))
        eog_raw.resample(work.info["sfreq"], npad="auto", verbose="ERROR")
        eog_types = {name: "eog" for name in eog_raw.ch_names}
        eog_raw.set_channel_types(eog_types, on_unit_change="ignore")

        combined = work.copy().add_channels([eog_raw], force_update_info=True)
        try:
            eog_inds = []
            for eog_name in eog_raw.ch_names:
                inds, scores = ica.find_bads_eog(
                    combined,
                    ch_name=eog_name,
                    threshold=3.0,
                    verbose="ERROR",
                )
                eog_inds.extend(inds)
            ica.exclude = sorted(set(eog_inds))
            if ica.exclude:
                warnings.warn(
                    f"ICA will exclude components {ica.exclude}; review their "
                    "topographies/time courses before accepting this cleaning."
                )
                ica.apply(work, verbose="ERROR")
        except Exception as exc:
            warnings.warn(f"ICA EOG scoring failed ({exc}); returning filtered EEG.")
        return work
