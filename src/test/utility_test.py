from pathlib import Path
import csv
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .lstm_classifier import LSTMClassifier


class Utility:
    @staticmethod
    def morlet_cwt(
        signal: np.ndarray, scales: np.ndarray, omega0: float = 6.0
    ) -> np.ndarray:
        coefficients = np.empty((len(scales), len(signal)), dtype=np.complex128)
        for index, scale in enumerate(scales):
            radius = min(max(2, int(np.ceil(4.0 * scale))), (len(signal) - 1) // 2)
            time = np.arange(-radius, radius + 1, dtype=np.float64)
            wavelet = (
                np.pi ** -0.25
                * np.exp(1j * omega0 * time / scale)
                * np.exp(-0.5 * (time / scale) ** 2)
                / np.sqrt(scale)
            )
            wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
            coefficients[index] = np.convolve(
                signal, np.conj(wavelet[::-1]), mode="same"
            )
        return coefficients

    @staticmethod
    def smooth_wavelet(
        values: np.ndarray, time_width: int = 5, scale_width: int = 3
    ) -> np.ndarray:
        time_kernel = np.ones(max(1, time_width), dtype=np.float64)
        time_kernel /= time_kernel.sum()
        smoothed = np.empty_like(values, dtype=np.complex128)
        for scale_index, row in enumerate(values):
            smoothed[scale_index] = np.convolve(row, time_kernel, mode="same")

        scale_kernel = np.ones(max(1, scale_width), dtype=np.float64)
        scale_kernel /= scale_kernel.sum()
        result = np.empty_like(smoothed, dtype=np.complex128)
        for time_index in range(smoothed.shape[1]):
            result[:, time_index] = np.convolve(
                smoothed[:, time_index], scale_kernel, mode="same"
            )
        return result

    @staticmethod
    def single_wavelet_coherence(
        real_signal: np.ndarray, fake_signal: np.ndarray
    ) -> float:
        if real_signal.shape != fake_signal.shape or real_signal.ndim != 1:
            raise ValueError(
                "Wavelet coherence requires two one-dimensional signals "
                "of equal length."
            )
        if not np.all(np.isfinite(real_signal)) or not np.all(
            np.isfinite(fake_signal)
        ):
            return float("nan")
        if (
            np.var(real_signal) <= np.finfo(float).eps
            or np.var(fake_signal) <= np.finfo(float).eps
        ):
            return float("nan")

        scales = np.geomspace(2.0, max(2.0, len(real_signal) / 4.0), num=24)
        real_cwt = Utility.morlet_cwt(real_signal - real_signal.mean(), scales)
        fake_cwt = Utility.morlet_cwt(fake_signal - fake_signal.mean(), scales)
        cross_spectrum = real_cwt * np.conj(fake_cwt)
        real_power = np.abs(real_cwt) ** 2
        fake_power = np.abs(fake_cwt) ** 2

        smoothed_cross = Utility.smooth_wavelet(cross_spectrum)
        smoothed_real_power = Utility.smooth_wavelet(
            real_power.astype(np.complex128)
        ).real
        smoothed_fake_power = Utility.smooth_wavelet(
            fake_power.astype(np.complex128)
        ).real
        denominator = smoothed_real_power * smoothed_fake_power
        coherence = np.abs(smoothed_cross) ** 2 / np.maximum(
            denominator, np.finfo(float).eps
        )
        coherence = np.clip(coherence.real, 0.0, 1.0)

        time = np.arange(len(real_signal))
        distance_from_edge = np.minimum(time, len(real_signal) - 1 - time)
        coi_mask = distance_from_edge[None, :] >= np.sqrt(2.0) * scales[:, None]
        valid = coi_mask & np.isfinite(coherence) & (denominator > np.finfo(float).eps)
        return float(np.mean(coherence[valid])) if np.any(valid) else float("nan")

    @staticmethod
    def wavelet_coherence_score(
        real: np.ndarray,
        fake: np.ndarray,
        channel_index: Optional[int] = None,
    ) -> float:
        if (
            real.shape != fake.shape
            or real.ndim != 3
            or real.shape[1] == 0
        ):
            raise ValueError("Expected matching arrays with shape (B, C, T), C > 0.")
        if channel_index is not None and not 0 <= channel_index < real.shape[1]:
            raise ValueError(
                f"channel_index must be between 0 and {real.shape[1] - 1}."
            )

        channels = range(real.shape[1]) if channel_index is None else (channel_index,)
        scores = [
            Utility.single_wavelet_coherence(
                real_sample[channel], fake_sample[channel]
            )
            for real_sample, fake_sample in zip(real, fake)
            for channel in channels
        ]
        valid_scores = np.asarray(scores, dtype=np.float64)
        valid_scores = valid_scores[np.isfinite(valid_scores)]
        return float(np.mean(valid_scores)) if valid_scores.size else float("nan")

    @staticmethod
    def discriminative_score(
        real: torch.Tensor,
        fake: torch.Tensor,
        device: torch.device,
        seed: int = 42,
        epochs: int = 30,
        minimum_real_samples: Optional[int] = None,
        bounded_split: bool = False,
        model_seed: Optional[int] = None,
        diagnostics_dir: Optional[Path] = None,
    ) -> float:
        """Score originale |test_accuracy - 0.5| con diagnostiche opzionali.

        L'accoppiamento real[i] / fake[i] preserva lo split senza leakage per
        il Signal-conditional. Nel Label-conditional e' soltanto stratificazione.
        """
        if len(real) != len(fake):
            raise ValueError("Discriminative Score requires equal real/fake sample counts.")
        if real.ndim != 3 or fake.shape != real.shape:
            raise ValueError("Expected matching EEG tensors shaped (B, C, T).")
        if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
            raise ValueError("Discriminative Score inputs contain non-finite values")
        diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
        min_samples = max(10, minimum_real_samples or 0)
        if len(real) < min_samples:
            if diagnostics_dir is not None:
                (diagnostics_dir / "summary.json").write_text(json.dumps({
                    "status": "insufficient_samples", "samples_per_origin": len(real),
                    "minimum_samples_per_origin": min_samples,
                    "discriminative_score": None
                }, indent=2), encoding="utf-8")
            return float("nan")

        features = torch.cat([real, fake], dim=0)
        targets = torch.cat([torch.ones(len(real)), torch.zeros(len(fake))], dim=0)

        # Split bilanciato, per coppie: mai originale e ricostruzione in gruppi diversi.
        n = len(real)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        split = max(1, min(n - 1, int(0.7 * n)))
        train_pairs, test_pairs = indices[:split], indices[split:]
        train_idx = torch.cat([train_pairs, train_pairs + n])
        test_idx = torch.cat([test_pairs, test_pairs + n])
        train_loader = DataLoader(
            TensorDataset(features[train_idx], targets[train_idx]),
            batch_size=32,
            shuffle=True,
        )

        if model_seed is not None:
            torch.manual_seed(model_seed)
        model = LSTMClassifier(input_size=real.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        def evaluate_split(indices_for_split: torch.Tensor) -> dict:
            """Valutazione senza backprop, in batch, solo per diagnostica."""
            model.eval()
            logits_list = []
            with torch.no_grad():
                for start in range(0, len(indices_for_split), 32):
                    ix = indices_for_split[start:start + 32]
                    logits_list.append(model(features[ix].to(device)).cpu().flatten())
            logits = torch.cat(logits_list)
            y = targets[indices_for_split].cpu().flatten()
            p = torch.sigmoid(logits)
            predicted = (p >= 0.5).to(y.dtype)
            tp = int(((predicted == 1) & (y == 1)).sum().item())
            tn = int(((predicted == 0) & (y == 0)).sum().item())
            fp = int(((predicted == 1) & (y == 0)).sum().item())
            fn = int(((predicted == 0) & (y == 1)).sum().item())
            recall_real = tp / (tp + fn) if tp + fn else float('nan')
            recall_fake = tn / (tn + fp) if tn + fp else float('nan')
            return {
                "loss": float(criterion(logits, y).item()),
                "accuracy": float((predicted == y).float().mean().item()),
                "balanced_accuracy": float((recall_real + recall_fake) / 2),
                "predicted_real_fraction": float(predicted.float().mean().item()),
                "prob_real_mean": float(p[y == 1].mean().item()),
                "prob_fake_mean": float(p[y == 0].mean().item()),
                "probability_std": float(p.std(unbiased=False).item()),
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                "n_real": int((y == 1).sum().item()),
                "n_fake": int((y == 0).sum().item()),
            }

        history = []
        model.train()
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            if diagnostics_dir is not None:
                train_stats = evaluate_split(train_idx)
                test_stats = evaluate_split(test_idx)
                history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_stats["loss"],
                    "test_loss": test_stats["loss"],
                    "train_accuracy": train_stats["accuracy"],
                    "test_accuracy": test_stats["accuracy"],
                    "train_balanced_accuracy": train_stats["balanced_accuracy"],
                    "test_balanced_accuracy": test_stats["balanced_accuracy"],
                    "train_predicted_real_fraction": train_stats["predicted_real_fraction"],
                    "test_predicted_real_fraction": test_stats["predicted_real_fraction"],
                })
                model.train()

        # La metrica rimane DEFINITA come nel programma originale: |accuracy - 0.5|.
        model.eval()
        with torch.no_grad():
            x_test = features[test_idx].to(device)
            y_test = targets[test_idx].to(device).unsqueeze(1)
            predictions = torch.sigmoid(model(x_test)) >= 0.5
            accuracy = (predictions == y_test).float().mean().item()
        score = float(abs(accuracy - 0.5))

        if diagnostics_dir is not None:
            train_stats = evaluate_split(train_idx)
            test_stats = evaluate_split(test_idx)
            warning_flags = []
            if train_stats["predicted_real_fraction"] in (0.0, 1.0):
                warning_flags.append("LSTM_predicts_only_one_class_on_training")
            if test_stats["predicted_real_fraction"] in (0.0, 1.0):
                warning_flags.append("LSTM_predicts_only_one_class_on_test")
            if train_stats["accuracy"] <= 0.55:
                warning_flags.append("train_accuracy_near_chance_or_worse")
            if train_stats["probability_std"] < 0.02:
                warning_flags.append("nearly_constant_predicted_probabilities")

            summary = {
                "status": "ok", "score_definition": "abs(test_accuracy - 0.5)",
                "discriminative_score": score, "seed": seed,
                "model_seed": model_seed, "epochs": epochs,
                "samples_per_origin": n,
                "train": train_stats, "test": test_stats,
                "warning_flags": warning_flags,
                "note": ("Flags are diagnostics, not proof of generator quality. "
                         "Near-chance accuracy could indicate a good generator or an "
                         "ineffective discriminator; use a separable-data positive control."),
            }
            with (diagnostics_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            (diagnostics_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # Grafici diagnostici, senza modificare il classificatore o la metrica.
            import matplotlib.pyplot as plt
            for quantity, label in (("loss", "BCE loss"), ("accuracy", "Accuracy")):
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot([row["epoch"] for row in history],
                        [row[f"train_{quantity}"] for row in history], label="Train")
                ax.plot([row["epoch"] for row in history],
                        [row[f"test_{quantity}"] for row in history], label="Test")
                if quantity == "accuracy":
                    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
                    ax.set_ylim(0, 1)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(label)
                ax.set_title(f"Discriminative Score - {label}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(diagnostics_dir / f"{quantity}.png", dpi=130)
                plt.close(fig)
            print(f"[DS] {diagnostics_dir}: test_acc={accuracy:.3f}, DS={score:.3f}, "
                  f"train_acc={train_stats['accuracy']:.3f}, "
                  f"predicted_real_fraction={test_stats['predicted_real_fraction']:.2f}, "
                  f"flags={warning_flags or 'none'}")
        return score

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        checkpoint: Optional[Path],
        device: torch.device,
        expected_class_id: Optional[int] = None,
        allow_missing: bool = False,
        allow_model_key: bool = False,
        allow_module_payload: bool = False,
    ) -> None:
        if checkpoint is None:
            if allow_missing:
                return
            raise ValueError("A checkpoint path is required.")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(payload, dict):
            checkpoint_class_id = payload.get("class_id")
            metadata = payload.get("args", {})
            if checkpoint_class_id is None and isinstance(metadata, dict):
                checkpoint_class_id = metadata.get("class_id")
            if expected_class_id is not None and checkpoint_class_id is None:
                raise ValueError(
                    f"Checkpoint privo di class_id: impossibile verificare la classe per {checkpoint}."
                )
            if (
                expected_class_id is not None
                and checkpoint_class_id is not None
                and int(checkpoint_class_id) != expected_class_id
            ):
                raise ValueError(
                    f"Checkpoint is trained for class {checkpoint_class_id}, "
                    f"but class {expected_class_id} was requested."
                )

            candidate_keys = ["model_state_dict", "state_dict"]
            if allow_model_key:
                candidate_keys.append("model")

            state = payload
            for key in candidate_keys:
                if key in payload:
                    state = payload[key]
                    break
        else:
            state = payload

        if allow_module_payload and isinstance(state, nn.Module):
            state = state.state_dict()
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint architecture does not match the requested model. "
                "This usually means the checkpoint was trained on a different "
                "dataset/configuration (for example different channels or class count). "
                f"Original error: {exc}"
            ) from exc
