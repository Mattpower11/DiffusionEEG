"""BCI Competition IV Dataset 2b preprocessing for BioDiffusion.

Training sessions: 01T, 02T, 03T. Evaluation sessions: 04E, 05E.
Outputs: separate train/test arrays, audit reports, and train-fitted normalization.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import csv
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.conf import EEG_preprocessing_conf
from src.preprocessing.bci2b_pipeline import BCI2bPreprocessor
from src.preprocessing.bci2b_normalization import BCI2bNormalization
from src.preprocessing.bci2b_dataset import BCI2bDataset
from src.preprocessing.bci2b_diagnostics import BCI2bDiagnostics

# Parenthesized suffixes are allowed for locally downloaded copies, e.g.
# B0104E(4).gdf -> canonical session B0104E.gdf for MATLAB label lookup.
GDF_RE = re.compile(r"^(B\d{2}0[1-5][TE])(?:\s*\(\d+\))?\.gdf$", re.IGNORECASE)
EXPECTED_TRAIN = {f"B{s:02d}0{k}T.gdf" for s in range(1, 10) for k in (1, 2, 3)}
EXPECTED_TEST = {f"B{s:02d}0{k}E.gdf" for s in range(1, 10) for k in (4, 5)}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def discover_sessions(data_dir: Path, pattern: str) -> Tuple[List[Path], List[Path], Dict[str, str]]:
    """Return train/test paths and exact-path -> canonical filename mapping."""
    if not data_dir.is_dir():
        raise NotADirectoryError(f"GDF data directory does not exist: {data_dir}")

    selected: Dict[str, Path] = {}
    canonical_for_path: Dict[str, str] = {}
    
    for path in sorted(data_dir.glob(pattern)):
        if not path.is_file():
            continue
        
        match = GDF_RE.fullmatch(path.name)
        
        if not match:
            # Other files in the directory are not BCI IV 2b sessions.
            continue
        
        canonical = match.group(1).upper() + ".gdf"
        
        if canonical not in EXPECTED_TRAIN and canonical not in EXPECTED_TEST:
            continue
        
        if canonical in selected:
            raise ValueError(
                f"Duplicate GDFs for {canonical}: {selected[canonical]} and {path}. "
                "Keep only one copy of each session."
            )
        
        selected[canonical] = path
        canonical_for_path[str(path)] = canonical

    train = [selected[name] for name in sorted(EXPECTED_TRAIN & selected.keys())]
    test = [selected[name] for name in sorted(EXPECTED_TEST & selected.keys())]
    
    return train, test, canonical_for_path


def verify_session_set(
    train: List[Path],
    test: List[Path],
    canonical_for_path: Dict[str, str],
    allow_partial: bool,
) -> None:
    train_names = {canonical_for_path[str(p)] for p in train}
    test_names = {canonical_for_path[str(p)] for p in test}
    missing_train = sorted(EXPECTED_TRAIN - train_names)
    missing_test = sorted(EXPECTED_TEST - test_names)
    
    if not train or not test:
        raise ValueError(
            f"Both train and evaluation GDFs are required: found {len(train)} train "
            f"and {len(test)} test."
        )
    
    if (missing_train or missing_test) and not allow_partial:
        raise FileNotFoundError(
            "Missing expected BCI IV 2b sessions: "
            f"{len(missing_train)} train, {len(missing_test)} test. "
            f"Train missing: {missing_train}; test missing: {missing_test}. "
            "Use --allow-partial only for deliberate subset experiments."
        )
    
    if missing_train or missing_test:
        warnings.warn(
            f"Partial dataset: {len(train)}/27 training and {len(test)}/18 "
            "evaluation GDFs. Results are not a full-dataset experiment.",
            stacklevel=2,
        )


def verify_labels_source(labels_path: Path, test: List[Path], canonical_for_path: Dict[str, str]) -> None:
    """Fail fast if any evaluation .mat file is absent (without reading labels here)."""
    expected = {Path(canonical_for_path[str(p)]).stem.lower() + ".mat" for p in test}
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Official label source not found: {labels_path}")
   
    if labels_path.is_dir():
        available = [p.name.lower() for p in labels_path.rglob("*.mat")]
    elif labels_path.is_file() and labels_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(labels_path) as archive:
            bad_entry = archive.testzip()
            
            if bad_entry is not None:
                raise ValueError(f"Corrupted ZIP entry: {bad_entry}")
           
            available = [Path(name).name.lower() for name in archive.namelist()
                         if name.lower().endswith(".mat") and not name.endswith("/")]
    elif labels_path.is_file() and labels_path.suffix.lower() == ".mat":
        available = [labels_path.name.lower()]
    else:
        raise ValueError("--labels-path must be an archive .zip, a directory or a .mat file")
    
    for name in sorted(expected):
        count = available.count(name)
        
        if count != 1:
            raise ValueError(f"Official labels: expected exactly one {name}; found {count} in {labels_path}")


def validate_arrays(name: str, X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.dtype != np.float32 or X.ndim != 3 or X.shape[1:] != (3, 500):
        raise ValueError(f"{name}: X must be float32 [N, 3, 500]; got {getattr(X, 'shape', None)}, {getattr(X, 'dtype', None)}")
   
    if not isinstance(y, np.ndarray) or y.dtype != np.int64 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{name}: y must be int64 [N] and match X; got {getattr(y, 'shape', None)}, {getattr(y, 'dtype', None)}")
    
    if not np.isfinite(X).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{name}: NaN/Inf or unexpected class labels")
    
    if not len(y):
        raise ValueError(f"{name}: no retained epochs")


def validate_report(path: Path, n_epochs: int, n_files: int, split: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{split}: missing aggregate preprocessing report {path}")
    
    report = json.loads(path.read_text(encoding="utf-8"))
    original = int(report["initial_original_trials"])
    kept = int(report["kept_trials"])
    rejected = int(report["rejected_trials"])
    
    if original != kept + rejected or kept != n_epochs or int(report["n_files"]) != n_files:
        raise ValueError(f"{split}: report totals disagree with the resulting dataset: {path}")
    
    if report.get("split") != split:
        raise ValueError(f"{split}: unexpected split in report: {report.get('split')!r}")

    audit_path = path.with_name("trial_audit.csv")
    if not audit_path.is_file():
        raise FileNotFoundError(f"{split}: missing trial audit {audit_path}")
    
    with audit_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    
    if len(rows) != original or sum(row["kept"].strip().lower() == "true" for row in rows) != kept:
        raise ValueError(f"{split}: CSV audit totals disagree with aggregate report")


def save_split(folder: Path, X: np.ndarray, y: np.ndarray, metadata: List[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "X.npy", X, allow_pickle=False)
    np.save(folder / "y.npy", y, allow_pickle=False)
    (folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_available_diagnostics(
    diagnostics: BCI2bDiagnostics,
    X: np.ndarray,
    y: np.ndarray,
    before: Optional[np.ndarray],
    config: EEG_preprocessing_conf,
    output_dir: Path,
    report_dir: Path,
) -> None:
    diagnostics.plot_diagnostics(X, y, sfreq=config.sfreq_expected, output_dir=output_dir)
    extras = (
        ("plot_psd_comparison", dict(X_before=before, X_after=X, sfreq=config.sfreq_expected,
                                      output_dir=output_dir)),
        ("plot_multiple_epochs", dict(X=X, y=y, sfreq=config.sfreq_expected,
                                      output_dir=output_dir, n_per_class=3,
                                      random_seed=config.random_seed)),
        ("plot_peak_to_peak_distribution", dict(X=X, threshold_uv=config.peak_to_peak_threshold_uv,
                                                output_dir=output_dir)),
        ("plot_trials_by_subject", dict(report_path=report_dir / "preprocessing_report.json",
                                        output_dir=output_dir)),
    )
    
    for method, kwargs in extras:
        fn = getattr(diagnostics, method, None)
        
        if not callable(fn):
            print(f"[plots] Optional diagnostic {method} not present: skipped")
            continue
        
        if method == "plot_psd_comparison" and before is None:
            print("[plots] Missing matched pre-filter epochs: PSD comparison skipped")
            continue
        
        fn(**kwargs)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing original BCI2b GDF files")
    parser.add_argument("--labels-path", type=Path, default=None,
                        help="Extracted MATLAB label directory (default: --data-dir/true_labels); also accepts a ZIP or one .mat")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_bci2b"))
    parser.add_argument("--pattern", type=str, default="*.gdf",
                        help="Glob for GDF discovery (legacy argument; defaults to all GDF files)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Permit missing subjects/sessions for a deliberate subset run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be >=1 and --num-workers must be >=0")

    labels_path = args.labels_path if args.labels_path is not None else args.data_dir / "true_labels"
    train_files, test_files, session_names = discover_sessions(args.data_dir, args.pattern)
    verify_session_set(train_files, test_files, session_names, args.allow_partial)
    verify_labels_source(labels_path, test_files, session_names)

    config = EEG_preprocessing_conf()
    config.labels_archive_path = str(labels_path)
    
    if not callable(getattr(config, "session_spec", None)):
        raise RuntimeError(
            "The installed src.conf is not the session-aware version. "
            "Install the updated EEG_preprocessing_conf.py as src/conf.py first."
        )
    
    if getattr(config, "normalization", None) != "train":
        raise ValueError("This train/test experiment requires config.normalization='train', not trial-wise fitting")
    
    if callable(getattr(config, "validate", None)):
        config.validate()

    train_split_names = {config.session_spec(session_names[str(p)]).split for p in train_files}
    test_split_names = {config.session_spec(session_names[str(p)]).split for p in test_files}
    
    if len(train_split_names) != 1 or len(test_split_names) != 1 or train_split_names == test_split_names:
        raise ValueError(
            f"Incompatible session split definitions: training={train_split_names}, "
            f"evaluation={test_split_names}. Check session-aware configuration."
        )
    
    train_split_name = next(iter(train_split_names))
    test_split_name = next(iter(test_split_names))

    normalizer = BCI2bNormalization(config)
    preprocessor = BCI2bPreprocessor(config)
    diagnostics = BCI2bDiagnostics()
    output = args.output_dir
    reports = output / "reports"
    
    print(f"Training files: {len(train_files)} | Evaluation files: {len(test_files)}")
    print(f"Official labels: {labels_path}")

    config.report_output_dir = str(reports / "train")
    X_train_raw, y_train, meta_train = preprocessor.preprocess_dataset(
        train_files, collect_unfiltered=args.plot, session_names=session_names
    )
    before_train = preprocessor.unfiltered_epochs if args.plot else None
    
    validate_arrays("train raw", X_train_raw, y_train)
    validate_report(reports / "train" / "preprocessing_report.json", len(y_train), len(train_files), train_split_name)

    # Fit once, on training EEG only. Never fit or update statistics on evaluation EEG.
    stats = normalizer.fit_normalization(X_train_raw, mode="train")
    
    if stats.get("mode") != "train":
        raise RuntimeError("Normalizer did not produce train-fitted statistics")
    
    X_train = normalizer.normalize_data(X_train_raw, stats)
    validate_arrays("train normalized", X_train, y_train)

    config.report_output_dir = str(reports / "test")
    X_test_raw, y_test, meta_test = preprocessor.preprocess_dataset(
        test_files, collect_unfiltered=args.plot, labels_path=labels_path,
        session_names=session_names
    )
    before_test = preprocessor.unfiltered_epochs if args.plot else None
    
    validate_arrays("test raw", X_test_raw, y_test)
    validate_report(reports / "test" / "preprocessing_report.json", len(y_test), len(test_files), test_split_name)
    
    X_test = normalizer.normalize_data(X_test_raw, stats)
    
    validate_arrays("test normalized", X_test, y_test)

    save_split(output / "train", X_train, y_train, meta_train)
    save_split(output / "test", X_test, y_test, meta_test)
   
    normalizer.save_normalization_parameters(stats, output / "normalization.json")

    print("\nTRAIN (normalized):")
    diagnostics.print_diagnostics(X_train, y_train)
    
    print("\nTEST (normalized using train stats):")
    diagnostics.print_diagnostics(X_test, y_test)

    # Dataset gets already normalized arrays: stats=None prevents normalization twice.
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        dataset = BCI2bDataset(X, y, stats=None, unconditional=False)
        loader = create_dataloader(dataset, batch_size=args.batch_size,
                                   shuffle=(split == "train"), num_workers=args.num_workers,
                                   pin_memory=False)
        xb, yb = next(iter(loader))
        print(f"{split} DataLoader batch: {tuple(xb.shape)}, {tuple(yb.shape)}")

    if args.plot:
        # Plot pre-normalization Volt signals (plots convert Volts -> microvolts).
        plot_available_diagnostics(diagnostics, X_train_raw, y_train, before_train,
                                   config, output / "train" / "plots", reports / "train")
        plot_available_diagnostics(diagnostics, X_test_raw, y_test, before_test,
                                   config, output / "test" / "plots", reports / "test")

    print(f"\nComplete. Output: {output.resolve()}")


if __name__ == "__main__":
    main()


'''
python src/preprocessing/BCI2b_preprocessing.py --data-dir ./datasets/BCICIV_2b_gdf/data --labels-path ./datasets/BCICIV_2b_gdf/true_labels --output-dir ./datasets/processed_BCICIV_2b_gdf --batch-size 64 --plot
'''