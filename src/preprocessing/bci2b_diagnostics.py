from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import welch


class BCI2bDiagnostics:
    @staticmethod
    def print_diagnostics(X: np.ndarray, y: np.ndarray) -> None:
        print(f"X shape: {X.shape} (N, channels, time)")
        print(f"y shape: {y.shape}")
        print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
        print("Finite:", bool(np.isfinite(X).all()))
        print("Per-channel mean:", X.mean(axis=(0, 2)))
        print("Per-channel std:", X.std(axis=(0, 2)))

    @staticmethod
    def plot_diagnostics(
        X: np.ndarray,
        y: np.ndarray,
        sfreq: float = 250.0,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Optional plots: trace, PSD, amplitude histograms, and class counts."""
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir or "eeg_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        t = np.arange(X.shape[-1]) / sfreq
        for ch, ax in enumerate(axes):
            ax.plot(t, X[0, ch] * 1e6)
            ax.set_ylabel(f"EEG {ch+1} (µV)")
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(output_dir / "example_epoch.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for ch in range(3):
            f, pxx = welch(
                X[:, ch, :].reshape(-1),
                fs=sfreq,
                nperseg=min(256, X.shape[-1]),
            )
            ax.semilogy(f, pxx, label=f"Channel {ch+1}")
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "psd.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ch, ax in enumerate(axes):
            ax.hist((X[:, ch, :] * 1e6).ravel(), bins=80)
            ax.set_title(f"Channel {ch+1}")
            ax.set_xlabel("Amplitude (µV)")
        fig.tight_layout()
        fig.savefig(output_dir / "amplitude_histograms.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]
        ax.bar(["Left hand", "Right hand"], counts)
        ax.set_ylabel("Epoch count")
        fig.tight_layout()
        fig.savefig(output_dir / "class_distribution.png", dpi=150)
        plt.close(fig)
