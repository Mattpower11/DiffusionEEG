"""
BCI Competition IV Dataset 2b preprocessing for BioDiffusion.

Expected output:
    X: float32 array/tensor [N, 3, sequence_length]
    y: int64 labels [N], 0=Left Hand, 1=Right Hand

Important:
- Inspect channel names and annotation/event mapping for every recording.
- The script does not assume channel order. EEG channel names are selected using
  an explicit configurable mapping/pattern list; ambiguous selections fail loudly.
- EOG channels are used only as auxiliary channels for optional ICA scoring.
- ICA with only 3 EEG channels is not recommended for robust ocular-component
  removal. The optional ICA mode fits ICA on EEG and uses EOG channels as
  auxiliary references; review components before trusting automated exclusion.
- Filtering is applied to continuous recordings before epoching to reduce edge
  artifacts. Epoch timing must be chosen from the official cue/trial protocol.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.conf import EEG_preprocessing_conf

import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.bci2b_pipeline import BCI2bPreprocessor
from src.preprocessing.bci2b_normalization import BCI2bNormalization
from src.preprocessing.bci2b_dataset import BCI2bDataset
from src.preprocessing.bci2b_diagnostics import BCI2bDiagnostics

CFG = EEG_preprocessing_conf()
NORMALIZATION_PREPROCESSOR = BCI2bNormalization(CFG)
DATA_PREPROCESSOR = BCI2bPreprocessor(CFG)
DIAGNOSTICS = BCI2bDiagnostics()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_bci2b"))
    parser.add_argument("--pattern", type=str, default="*0[12]T.gdf",
                        help="Use labelled screening sessions 01T/02T by default.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.data_dir}")

    X, y, metadata = DATA_PREPROCESSOR.preprocess_dataset(files)
    DIAGNOSTICS.print_diagnostics(X, y)

    # IMPORTANT: this script treats the supplied files as the training split.
    # For rigorous experiments, split by subject/session BEFORE fitting stats.
    stats = NORMALIZATION_PREPROCESSOR.fit_normalization(X)
    X_norm = NORMALIZATION_PREPROCESSOR.normalize_data(X, stats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X.npy", X_norm)
    np.save(args.output_dir / "y.npy", y)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    NORMALIZATION_PREPROCESSOR.save_normalization_parameters(
        stats,
        args.output_dir / "normalization.json",
    )

    dataset = BCI2bDataset(X, y, stats=stats, unconditional=False)
    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    xb, yb = next(iter(loader))
    print("DataLoader batch:", tuple(xb.shape), tuple(yb.shape))

    if args.plot:
        DIAGNOSTICS.plot_diagnostics(
            X,
            y,
            sfreq=CFG.sfreq_expected,
            output_dir=args.output_dir / "plots",
        )


if __name__ == "__main__":
    main()

'''
python src/preprocessing/BCI2b_preprocessing.py --data-dir ./datasets/BCICIV_2b_gdf --output-dir ./datasets/processed_BCICIV_2b_gdf --batch-size 64 --plot
'''