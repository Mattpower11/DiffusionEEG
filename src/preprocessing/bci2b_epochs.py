"""BCI Competition IV 2b: cue-aligned, trial-index-stable EEG epoch extraction.

Depends on EEG_preprocessing_conf(6).py (install as src/conf.py).
Evaluation labels are obtained *before rejection* from the matching Bxx04E.mat / Bxx05E.mat.
The returned X/y include only kept epochs, in original trial order.
"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import warnings
import zipfile

import mne
import numpy as np
from scipy.io import loadmat

from src.conf import BCI2bSessionSpec, EEG_preprocessing_conf


class BCI2bEpochPreprocessor:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config
        self.last_trial_report = []
        self.last_session_spec = None
        self.last_label_source = None

    @staticmethod
    def _event_table(raw_full):
        events, event_id = mne.events_from_annotations(raw_full, verbose="ERROR")
        return events, {str(k).strip(): int(v) for k, v in event_id.items()}

    def get_motor_imagery_events(self, raw_full) -> Tuple[np.ndarray, Dict[str, int]]:
        """Backward-compatible helper for labeled training (769/770) events."""
        events, ids = self._event_table(raw_full)
        for name in ("769", "770"):
            if name not in ids:
                raise ValueError(f"Expected annotation {name}; available {sorted(ids)}")
        return events, {"left_hand": ids["769"], "right_hand": ids["770"]}

    def _load_external_labels(self, filename: str, source: Optional[Union[str, Path]]):
        if source is None:
            raise ValueError(
                f"{filename}: evaluation labels are required. Supply labels_path=... "
                "or set config.labels_archive_path to true_labels.zip or its directory."
            )
        path = Path(source)
        matlab_name = f"{Path(filename).stem}.mat"
        
        if path.is_dir():
            matches = [p for p in path.rglob('*.mat') if p.name.lower() == matlab_name.lower()]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {matlab_name} under {path}; found {len(matches)}")
            mat = loadmat(str(matches[0]))
            actual_source = str(matches[0])
        elif path.is_file() and path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                matches = [name for name in archive.namelist()
                           if Path(name).name.lower() == matlab_name.lower() and not name.endswith('/')]
                
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {matlab_name} in {path}; found {len(matches)}")
                
                mat = loadmat(BytesIO(archive.read(matches[0])))
                actual_source = f"{path}!{matches[0]}"
        elif path.is_file() and path.suffix.lower() == '.mat':
            if path.name.lower() != matlab_name.lower():
                raise ValueError(f"Label file {path.name} does not match session {matlab_name}")
            
            mat = loadmat(str(path))
            actual_source = str(path)
        else:
            raise ValueError(f"Label source not found/unsupported: {path}")
       
        variable = self.config.labels_variable_name
        if variable not in mat:
            raise ValueError(f"{actual_source}: missing MATLAB variable {variable!r}")
        
        raw = np.asarray(mat[variable])
        
        if raw.ndim not in (1, 2) or (raw.ndim == 2 and min(raw.shape) != 1):
            raise ValueError(f"{actual_source}: classlabel must be a vector; got {raw.shape}")
        
        values = raw.reshape(-1, order='F')
        
        if values.dtype.kind not in 'iuf' or not np.all(np.isfinite(values)):
            raise ValueError(f"{actual_source}: classlabel must contain finite numeric values")
        
        if not np.all(np.isin(values, [self.config.matlab_left_hand_label,
                                     self.config.matlab_right_hand_label])):
            raise ValueError(f"{actual_source}: unexpected classlabel values {np.unique(values)}")
        
        labels = np.where(values == self.config.matlab_left_hand_label,
                          self.config.output_left_hand_label,
                          self.config.output_right_hand_label).astype(np.int64)
        
        return labels, actual_source

    def extract_epochs(
        self,
        raw_eeg: mne.io.BaseRaw,
        raw_full: mne.io.BaseRaw,
        *,
        file_path: Optional[Union[str, Path]] = None,
        labels_path: Optional[Union[str, Path]] = None,
        external_labels: Optional[np.ndarray] = None,
        session_spec: Optional[BCI2bSessionSpec] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Extract X [N, 3, T], y [N] with stable trial indices and audit.

        file_path is required if the original filename is unavailable on raw_full.
        labels_path can refer to a matching .mat file, containing directory, or ZIP.
        external_labels, if supplied, are original MATLAB values (1/2), unfiltered.
        """
        self.last_trial_report = []
        self.last_session_spec = None
        self.last_label_source = None
        
        if file_path is None:
            filenames = getattr(raw_full, 'filenames', None) or []
            file_path = next((str(p) for p in filenames if p is not None), None)
        
        if session_spec is None:
            if file_path is None:
                raise ValueError('Cannot determine session: provide file_path or session_spec')
            spec = self.config.session_spec(file_path)
        else:
            spec = session_spec
        
        self.last_session_spec = spec
        self.config.validate()
        filename = Path(file_path).name if file_path is not None else f'B{spec.subject_id:02d}{spec.session_id}.gdf'
        
        if file_path is not None:
            expected_spec = self.config.session_spec(file_path)
            
            if (expected_spec.subject_id, expected_spec.session_id) != (spec.subject_id, spec.session_id):
                raise ValueError('session_spec does not match file_path')

        sfreq = float(raw_eeg.info['sfreq'])
        
        if not np.isclose(sfreq, float(raw_full.info['sfreq']), atol=1e-8, rtol=0):
            raise ValueError('raw_eeg and raw_full have different sampling frequencies')
        
        if raw_eeg.first_samp != raw_full.first_samp or raw_eeg.n_times != raw_full.n_times:
            raise ValueError('raw_eeg and raw_full have incompatible sample axes')
        
        if len(raw_eeg.ch_names) != 3 or any(t != 'eeg' for t in raw_eeg.get_channel_types()):
            raise ValueError('raw_eeg must have exactly three EEG channels, without EOG')

        events, ids = self._event_table(raw_full)
        start_desc = self.config.trial_start_annotation
        
        if start_desc not in ids:
            raise ValueError(f'{filename}: missing trial-start marker {start_desc}')
        
        start_code = ids[start_desc]
        starts = np.sort(events[events[:, 2] == start_code, 0])
        
        if not len(starts):
            raise ValueError(f'{filename}: no trial starts')
        
        rejected_code = ids.get(self.config.rejected_trial_annotation)
        expert = events[events[:, 2] == rejected_code, 0] if rejected_code is not None else np.empty(0, dtype=int)
        
        if any(desc not in ids for desc in spec.cue_annotations):
            missing = [desc for desc in spec.cue_annotations if desc not in ids]
            raise ValueError(f'{filename}: missing cue annotation(s): {missing}')
        
        cue_codes = {desc: ids[desc] for desc in spec.cue_annotations}
        cues = events[np.isin(events[:, 2], list(cue_codes.values()))]
        feedback_code = ids.get(self.config.feedback_onset_annotation)
        feedback_samples = (events[events[:, 2] == feedback_code, 0]
                            if feedback_code is not None else np.empty(0, dtype=int))

        # Index trials by 768, NOT by filtered cues/epochs: indices never shift.
        trials = []
        for i, start in enumerate(starts):
            stop = int(starts[i+1]) if i+1 < len(starts) else np.iinfo(np.int64).max
            in_trial = cues[(cues[:, 0] >= start) & (cues[:, 0] < stop)]
            bad_marker = bool(np.any((expert >= start) & (expert < stop)))
            fb = feedback_samples[(feedback_samples >= start) & (feedback_samples < stop)]
            cue_sample = int(in_trial[0, 0]) if len(in_trial) == 1 else None
            trials.append(dict(trial_index=i, trial_id=i+1, start_sample=int(start),
                               cue_sample=cue_sample,
                               cue_code=int(in_trial[0, 2]) if len(in_trial) == 1 else None,
                               n_cues=int(len(in_trial)), n_feedback_events=int(len(fb)),
                               feedback_sample=int(fb[0]) if len(fb) == 1 else None,
                               expert_rejected_1023=bad_marker))
        # Cue outside any 768-delimited trial is an inconsistency, not a new trial.
        if any(int(cue[0]) < int(starts[0]) for cue in cues):
            raise ValueError(f'{filename}: cue precedes first trial-start marker')
        
        if spec.labels_source == 'external_mat':
            if external_labels is not None and labels_path is not None:
                raise ValueError('Provide external_labels or labels_path, not both')
            
            if external_labels is not None:
                values = np.asarray(external_labels).reshape(-1)
                
                if values.dtype.kind not in 'iuf' or not np.all(np.isfinite(values)) or not np.all(
                    np.isin(values, [self.config.matlab_left_hand_label,
                                    self.config.matlab_right_hand_label])
                ):
                    raise ValueError('external_labels must contain MATLAB values 1/2')
                
                labels = np.where(values == self.config.matlab_left_hand_label, 0, 1).astype(np.int64)
                self.last_label_source = 'external_labels array'
            else:
                labels, source = self._load_external_labels(filename,
                    labels_path if labels_path is not None else self.config.labels_archive_path)
                self.last_label_source = source
            
            if len(labels) != len(trials):
                raise ValueError(f'{filename}: label/trial mismatch: {len(labels)} MATLAB labels '
                                 f'but {len(trials)} trial-start events 768. '
                                 'Labels must correspond to ALL original trials, before rejection.')
            
            for trial, label in zip(trials, labels):
                trial['label'] = int(label)
        else:
            # Training: require one known cue in every trial to avoid silent label shifts.
            malformed = [t['trial_id'] for t in trials if t['n_cues'] != 1]
            
            if malformed:
                raise ValueError(f'{filename}: training trials with missing/duplicate cues: {malformed}')
            
            left_desc, right_desc = spec.cue_annotations
            
            for trial in trials:
                trial['label'] = 0 if trial['cue_code'] == cue_codes[left_desc] else 1
            
            self.last_label_source = 'GDF 769/770'

        # Pre-rejection reference: cue onset, never event 768 or event 781.
        # 781 is a feedback onset, used for audit but not reinterpreted as a cue.
        candidate = []
        for trial in trials:
            if trial['cue_sample'] is not None and not trial['expert_rejected_1023']:
                candidate.append([trial['cue_sample'], 0, 1])
        candidate = np.array(candidate, dtype=np.int64).reshape(-1, 3)
        if len(candidate) != len({int(row[0]) for row in candidate}):
            raise ValueError(f'{filename}: duplicate cue sample positions')

        data_by_sample = {}
        good_annotation_samples = set()
        unavailable_samples = set()
        actual_t = self.config.expected_epoch_samples(spec.tmin, spec.tmax)
        if self.config.sequence_length is not None and actual_t != self.config.sequence_length:
            raise ValueError(f'{filename}: {spec.protocol} window yields {actual_t} samples, '
                             f'expected {self.config.sequence_length}; check session timings')
        if len(candidate):
            epochs_all = mne.Epochs(raw_eeg, candidate, event_id={'cue': 1},
                tmin=spec.tmin, tmax=spec.tmax, baseline=None, preload=True,
                reject_by_annotation=False, reject=None, flat=None, verbose='ERROR')
            epochs_bad = mne.Epochs(raw_eeg, candidate, event_id={'cue': 1},
                tmin=spec.tmin, tmax=spec.tmax, baseline=None, preload=True,
                reject_by_annotation=True, reject=None, flat=None, verbose='ERROR')
            epoch_data = epochs_all.get_data(copy=True)
            actual_t = len(epochs_all.times)
            
            if epoch_data.shape[-1] != actual_t:
                raise RuntimeError(f'{filename}: extracted time axis inconsistent with MNE epoch data')
            
            if self.config.sequence_length is not None and actual_t != self.config.sequence_length:
                raise ValueError(f'{filename}: MNE produced {actual_t} samples, '
                                 f'expected {self.config.sequence_length}')
            
            data_by_sample = {int(e[0]): epoch_data[i] for i, e in enumerate(epochs_all.events)}
            good_annotation_samples = set(int(e[0]) for e in epochs_bad.events)
            unavailable_samples = set(map(int, candidate[:, 0])) - set(data_by_sample)

        kept_data, kept_labels = [], []
        records = []
        for trial in trials:
            sample = trial['cue_sample']
            label = trial['label']
            reasons = []
            
            if sample is None:
                reasons.append('missing_or_multiple_cue')
            
            if trial['expert_rejected_1023']:
                reasons.append('expert_marker_1023')
           
            if sample in unavailable_samples:
                reasons.append('epoch_out_of_bounds_or_unavailable')
            
            epoch = data_by_sample.get(sample) if sample is not None else None
            annotation_bad = None
            max_abs = None
            p2p_values = {}
            flat_channels = []
            
            if epoch is not None:
                annotation_bad = sample not in good_annotation_samples
                
                if annotation_bad:
                    reasons.append('bad_annotation_overlap')
                
                if not np.isfinite(epoch).all():
                    reasons.append('non_finite_values')
                else:
                    abs_uv = np.max(np.abs(epoch), axis=1) * 1e6
                    p2p = np.ptp(epoch, axis=1) * 1e6
                    max_abs = float(abs_uv.max())
                    p2p_values = {name: float(p2p[k]) for k, name in enumerate(raw_eeg.ch_names)}
                    flat_channels = [name for k, name in enumerate(raw_eeg.ch_names)
                                     if p2p[k] < self.config.flat_threshold_uv]
                    
                    if flat_channels:
                        reasons.append('flat_threshold')
                    
                    if np.any(p2p > self.config.peak_to_peak_threshold_uv):
                        reasons.append('peak_to_peak_threshold')
            record = {
                'trial_id': trial['trial_id'], 'trial_index': trial['trial_index'],
                'trial_start_sample': trial['start_sample'],
                'event_sample': sample if sample is not None else trial['start_sample'],
                'event_time_seconds': ((sample if sample is not None else trial['start_sample'])
                                       - raw_full.first_samp) / sfreq,
                'cue_sample': sample, 'feedback_sample': trial['feedback_sample'],
                'n_feedback_events': trial['n_feedback_events'],
                'protocol': spec.protocol, 'split': spec.split,
                'label_source': self.last_label_source,
                'label': int(label), 'class': 'left_hand' if label == 0 else 'right_hand',
                'expert_rejected_1023': trial['expert_rejected_1023'],
                'annotation_bad': annotation_bad, 'max_abs_amplitude_uv': max_abs,
                'peak_to_peak_uv': p2p_values, 'flat_channels': flat_channels,
                'kept': not reasons, 'rejection_reasons': reasons,
            }
            
            records.append(record)
            
            if not reasons:
                if epoch is None:
                    raise RuntimeError(f'{filename}: trial {trial["trial_id"]} has no epoch but was marked kept')
                
                kept_data.append(epoch)
                kept_labels.append(label)
        self.last_trial_report = records
        
        if not kept_data:
            raise ValueError(f'{filename}: all {len(trials)} trials rejected; '
                             f'reasons={dict(Counter(reason for r in records for reason in r["rejection_reasons"]))}')
        
        X = np.stack(kept_data).astype(np.float32)
        y = np.asarray(kept_labels, dtype=np.int64)
        
        if X.shape != (len(y), 3, actual_t) or not np.isfinite(X).all():
            raise RuntimeError(f'{filename}: invalid X/y tensor shape or nonfinite output')
        
        if not np.array_equal(y, np.array([r['label'] for r in records if r['kept']], dtype=np.int64)):
            raise RuntimeError(f'{filename}: epoch/label alignment inconsistency')
        
        counts = Counter(reason for record in records for reason in record['rejection_reasons'])
        
        if counts:
            warnings.warn(f'{filename}: rejected {len(records)-len(y)}/{len(records)} '
                          f'trials; reasons={dict(counts)}', stacklevel=2)
        
        # Codes describe source GDF markers. Evaluation has no class-specific event IDs.
        codes = ({'left_hand': cue_codes[spec.cue_annotations[0]],
                  'right_hand': cue_codes[spec.cue_annotations[1]]}
                 if spec.labels_source == 'gdf'
                 else {'unknown_cue': cue_codes[spec.cue_annotations[0]]})
        
        return X, y, codes
