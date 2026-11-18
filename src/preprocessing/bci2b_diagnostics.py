from pathlib import Path
from typing import Optional

import json
import re

import matplotlib.pyplot as plt
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
            f, psd_epochs = welch(
                X[:, ch, :],
                fs=sfreq,
                nperseg=min(500, X.shape[-1]),
                axis=-1
            )

            psd_mean = np.mean(psd_epochs, axis=0)
            ax.semilogy(f, psd_mean, label=f"Channel {ch+1}")
        
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

    @staticmethod
    def plot_psd_comparison(
        X_before: np.ndarray,
        X_after: np.ndarray,
        sfreq: float = 250.0,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Compare PSD before/after filtering on identical EEG epochs."""

        if X_before.shape != X_after.shape:
            raise ValueError(
                f"Shape mismatch: {X_before.shape} vs {X_after.shape}"
            )

        if X_before.ndim != 3 or X_before.shape[1] != 3:
            raise ValueError("Expected shape (N, 3, T).")

        if len(X_before) == 0:
            raise ValueError("No epochs available for PSD comparison.")

        if not (
            np.isfinite(X_before).all()
            and np.isfinite(X_after).all()
        ):
            raise ValueError("PSD inputs contain NaN or Inf.")

        output_dir = Path(output_dir or "eeg_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        channel_names = ["C3", "Cz", "C4"]
        fig, axes = plt.subplots(
            3, 1,
            figsize=(11, 10),
            sharex=True,
            sharey=True
        )

        for ch, ax in enumerate(axes):

            nperseg = min(500, X_before.shape[-1])

            freqs, psd_before = welch(
                X_before[:, ch, :],
                fs=sfreq,
                nperseg=nperseg,
                axis=-1,
            )

            _, psd_after = welch(
                X_after[:, ch, :],
                fs=sfreq,
                nperseg=nperseg,
                axis=-1,
            )

            mean_before = np.mean(psd_before, axis=0)
            mean_after = np.mean(psd_after, axis=0)

            # Convert PSD from V²/Hz to µV²/Hz.
            mean_before *= 1e12
            mean_after *= 1e12

            eps = np.finfo(float).tiny

            ax.semilogy(
                freqs,
                np.maximum(mean_before, eps),
                label="Before filtering",
                linewidth=1.6,
            )

            ax.semilogy(
                freqs,
                np.maximum(mean_after, eps),
                label="After filtering",
                linewidth=1.6,
            )

            ax.axvline(
                40,
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="Low-pass cutoff (40 Hz)"
            )

            ax.set_title(channel_names[ch])
            ax.set_ylabel("PSD (µV²/Hz)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Frequency (Hz)")
        axes[-1].set_xlim(0, min(80, sfreq / 2))

        fig.suptitle(
            "EEG Power Spectral Density - Before vs After Filtering"
        )

        fig.tight_layout()
        fig.savefig(
            output_dir / "psd_before_after.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)


    @staticmethod
    def plot_multiple_epochs(
        X: np.ndarray,
        y: np.ndarray,
        sfreq: float = 250.0,
        output_dir: Optional[Path] = None,
        n_per_class: int = 3,
        random_seed: int = 42,
    ) -> None:
        """
        Plot random and high-amplitude EEG epochs.

        X: (N, 3, T), EEG in Volts before normalization.
        y: (N,), labels 0=Left Hand, 1=Right Hand.

        Outputs:
        - random_epochs.png
        - extreme_epochs.png
        """

        if X.ndim != 3 or X.shape[1] != 3:
            raise ValueError("Expected X with shape (N, 3, T).")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf.")

        if n_per_class < 1:
            raise ValueError("n_per_class must be positive.")

        output_dir = Path(output_dir or "eeg_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(random_seed)

        channel_names = ["C3", "Cz", "C4"]
        class_names = {0: "Left Hand", 1: "Right Hand"}

        # Peak-to-peak amplitude for each channel.
        p2p = np.ptp(X, axis=-1) * 1e6

        # Maximum peak-to-peak across the three channels.
        max_p2p = np.max(p2p, axis=1)

        random_indices = []
        extreme_indices = []

        for label in [0, 1]:

            class_indices = np.where(y == label)[0]

            if len(class_indices) == 0:
                raise ValueError(
                    f"No epochs found for class {label}."
                )

            n_select = min(n_per_class, len(class_indices))

            # Random selection without replacement.
            selected_random = rng.choice(
                class_indices,
                size=n_select,
                replace=False
            )

            random_indices.extend(selected_random.tolist())

            # Select the highest peak-to-peak epochs.
            sorted_indices = class_indices[
                np.argsort(max_p2p[class_indices])[::-1]
            ]

            extreme_indices.extend(
                sorted_indices[:n_select].tolist()
            )

        def create_figure(indices, title, filename):

            n_rows = len(indices)
            fig, axes = plt.subplots(
                n_rows,
                3,
                figsize=(16, 2.6 * n_rows),
                sharex=True,
                sharey=True,
                squeeze=False
            )

            t = np.arange(X.shape[-1]) / sfreq

            # Shared amplitude limits across all displayed epochs.
            amplitude_uv = X[indices] * 1e6

            limit = float(np.max(np.abs(amplitude_uv)))

            if limit == 0:
                limit = 1.0

            limit *= 1.1

            for row, idx in enumerate(indices):

                for ch in range(3):

                    ax = axes[row, ch]

                    signal_uv = X[idx, ch] * 1e6

                    ax.plot(
                        t,
                        signal_uv,
                        linewidth=1.0
                    )

                    ax.set_ylim(-limit, limit)
                    ax.grid(True, alpha=0.25)

                    ax.set_title(
                        f"{channel_names[ch]} | "
                        f"P2P: {p2p[idx, ch]:.2f} µV",
                        fontsize=10
                    )

                    if ch == 0:
                        ax.set_ylabel(
                            f"Epoch {idx}\n"
                            f"{class_names[int(y[idx])]}\n"
                            f"Amplitude (µV)"
                        )

                    if row == n_rows - 1:
                        ax.set_xlabel("Time (s)")

            fig.suptitle(
                title,
                fontsize=15,
                fontweight="bold"
            )

            fig.tight_layout(rect=[0, 0, 1, 0.97])

            fig.savefig(
                output_dir / filename,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close(fig)

        create_figure(
            random_indices,
            "Random EEG Epochs",
            "random_epochs.png"
        )

        create_figure(
            extreme_indices,
            "Highest Peak-to-Peak EEG Epochs",
            "extreme_epochs.png"
        )

    
    @staticmethod
    def plot_peak_to_peak_distribution(
        X: np.ndarray,
        threshold_uv: float = 200.0,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Plot peak-to-peak amplitude distributions for EEG epochs.

        X: EEG signals in Volts, shape (N, 3, T).
        threshold_uv: epoch rejection threshold in microvolts.

        Output:
        - peak_to_peak_distribution.png
        """

        if X.ndim != 3 or X.shape[1] != 3:
            raise ValueError(
                f"Expected X with shape (N, 3, T), got {X.shape}"
            )

        if len(X) == 0:
            raise ValueError("No EEG epochs available.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf.")

        if not np.isfinite(threshold_uv) or threshold_uv <= 0:
            raise ValueError("Invalid peak-to-peak threshold.")

        output_dir = Path(output_dir or "eeg_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        channel_names = ["C3", "Cz", "C4"]

        # Peak-to-peak for every epoch and channel.
        # Shape: (N, 3), values in microvolts.
        p2p = np.ptp(X, axis=-1) * 1e6

        fig, axes = plt.subplots(
            1, 2,
            figsize=(15, 5)
        )

        # -----------------------------------------
        # 1. Box plot per channel
        # -----------------------------------------

        ax = axes[0]

        ax.boxplot(
            [p2p[:, ch] for ch in range(3)],
            tick_labels=channel_names,
            showfliers=True,
            patch_artist=True,
        )

        ax.axhline(
            threshold_uv,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Rejection threshold ({threshold_uv:g} µV)"
        )

        ax.set_title("Peak-to-Peak Amplitude by Channel")
        ax.set_xlabel("EEG Channel")
        ax.set_ylabel("Peak-to-Peak Amplitude (µV)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        # -----------------------------------------
        # 2. Histogram per channel
        # -----------------------------------------

        ax = axes[1]

        bins = np.histogram_bin_edges(
            p2p.ravel(),
            bins=50
        )

        for ch in range(3):
            ax.hist(
                p2p[:, ch],
                bins=bins,
                alpha=0.4,
                label=channel_names[ch],
            )

        ax.axvline(
            threshold_uv,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Rejection threshold ({threshold_uv:g} µV)"
        )

        ax.set_title("Peak-to-Peak Amplitude Distribution")
        ax.set_xlabel("Peak-to-Peak Amplitude (µV)")
        ax.set_ylabel("Epoch Count")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        fig.suptitle(
            "EEG Peak-to-Peak Distribution",
            fontsize=15,
            fontweight="bold"
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / "peak_to_peak_distribution.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        # -----------------------------------------
        # 3. Numerical diagnostics
        # -----------------------------------------

        print("\n--- PEAK-TO-PEAK DIAGNOSTICS ---")

        for ch, name in enumerate(channel_names):

            values = p2p[:, ch]

            print(f"\nChannel {name}")
            print(f"  Mean:   {np.mean(values):.3f} µV")
            print(f"  Median: {np.median(values):.3f} µV")
            print(f"  Std:    {np.std(values):.3f} µV")
            print(f"  Min:    {np.min(values):.3f} µV")
            print(f"  Max:    {np.max(values):.3f} µV")
            print(f"  P95:    {np.percentile(values, 95):.3f} µV")
            print(f"  P99:    {np.percentile(values, 99):.3f} µV")

            n_exceeded = np.sum(values > threshold_uv)

            print(
                f"  Above threshold: {n_exceeded}/{len(values)}"
            )

        print("--------------------------------\n")


    @staticmethod
    def plot_trials_by_subject(
        report_path: Path,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Plot retained/rejected EEG trials for each subject.

        Reads the preprocessing_report.json generated by
        BCI2bPreprocessor.

        Output:
            trials_by_subject.png
        """

        report_path = Path(report_path)

        if not report_path.exists():
            raise FileNotFoundError(
                f"Preprocessing report not found: {report_path}"
            )

        output_dir = Path(output_dir or "eeg_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        # Aggregate different sessions of the same subject.
        subjects = {}

        for record in report["files"]:

            filename = record["file"]

            # Example: B0101T.gdf -> B01
            match = re.match(
                r"^(B\d{2})\d{2}[TE]\.gdf$",
                filename,
                re.IGNORECASE,
            )

            if match is None:
                raise ValueError(
                    f"Unexpected BCI2b filename: {filename}"
                )

            subject = match.group(1).upper()

            summary = record["trial_summary"]

            initial = int(summary["initial_cued_trials"])
            kept = int(summary["kept_trials"])
            rejected = int(summary["rejected_trials"])

            if initial != kept + rejected:
                raise ValueError(
                    f"Inconsistent trial counts in {filename}: "
                    f"{initial} != {kept} + {rejected}"
                )

            if subject not in subjects:
                subjects[subject] = {
                    "initial": 0,
                    "kept": 0,
                    "rejected": 0,
                }

            subjects[subject]["initial"] += initial
            subjects[subject]["kept"] += kept
            subjects[subject]["rejected"] += rejected

        if not subjects:
            raise ValueError("No subject data found in report.")

        subject_names = sorted(subjects.keys())

        kept_counts = np.array([
            subjects[s]["kept"] for s in subject_names
        ])

        rejected_counts = np.array([
            subjects[s]["rejected"] for s in subject_names
        ])

        initial_counts = kept_counts + rejected_counts

        retention_rates = (
            100.0 * kept_counts / initial_counts
        )

        # --------------------------------------
        # Stacked bar chart
        # --------------------------------------

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(subject_names))

        bar_width = 0.7

        ax.bar(
            x,
            kept_counts,
            bar_width,
            label="Retained",
            color="tab:green",
        )

        ax.bar(
            x,
            rejected_counts,
            bar_width,
            bottom=kept_counts,
            label="Rejected",
            color="tab:red",
        )

        # Display retention percentage above each bar.
        for i in range(len(subject_names)):

            ax.text(
                x[i],
                initial_counts[i] + 3,
                f"{retention_rates[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(subject_names)

        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of Trials")

        ax.set_title(
            "BCI2b - Retained and Rejected Trials by Subject"
        )

        ax.set_ylim(
            0,
            max(initial_counts) * 1.15
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        ax.set_axisbelow(True)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir / "trials_by_subject.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        # --------------------------------------
        # Numerical summary
        # --------------------------------------

        print("\n--- TRIAL RETENTION BY SUBJECT ---")

        for i, subject in enumerate(subject_names):

            print(
                f"{subject}: "
                f"initial={initial_counts[i]}, "
                f"retained={kept_counts[i]}, "
                f"rejected={rejected_counts[i]}, "
                f"retention={retention_rates[i]:.2f}%"
            )

        print("----------------------------------\n")

