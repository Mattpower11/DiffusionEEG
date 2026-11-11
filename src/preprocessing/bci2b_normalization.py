import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.conf import EEG_preprocessing_conf


class BCI2bNormalization:
    def __init__(self, config: EEG_preprocessing_conf):
        self.config = config

    def fit_normalization(
        self,
        train_data: np.ndarray,
        mode: Optional[str] = None,
    ) -> Dict[str, object]:
        if mode is None:
            mode = self.config.normalization
        if mode == "trial":
            return {"mode": "trial", "eps": self.config.eps}
        if mode != "train":
            raise ValueError("normalization must be 'train' or 'trial'")
        mean = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True)
        std = np.maximum(std, self.config.eps)
        return {
            "mode": "train",
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "eps": self.config.eps,
        }

    def normalize_data(
        self,
        data: np.ndarray,
        stats: Dict[str, object],
    ) -> np.ndarray:
        if stats["mode"] == "trial":
            mean = data.mean(axis=-1, keepdims=True)
            std = np.maximum(data.std(axis=-1, keepdims=True), float(stats["eps"]))
            out = (data - mean) / std
        else:
            mean = np.asarray(stats["mean"])
            std = np.asarray(stats["std"])
            out = (data - mean) / std
        if not np.isfinite(out).all():
            raise ValueError("Normalized data contain NaN or Inf.")
        return out.astype(np.float32)

    def save_normalization_parameters(
        self,
        stats: Dict[str, object],
        path: Path,
    ) -> None:
        payload = {"mode": stats["mode"], "eps": stats["eps"]}
        if stats["mode"] == "train":
            payload["mean"] = np.asarray(stats["mean"]).reshape(-1).tolist()
            payload["std"] = np.asarray(stats["std"]).reshape(-1).tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_normalization_parameters(self, path: Path) -> Dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["mode"] == "train":
            payload["mean"] = np.asarray(
                payload["mean"],
                dtype=np.float32,
            ).reshape(1, 3, 1)
            payload["std"] = np.asarray(
                payload["std"],
                dtype=np.float32,
            ).reshape(1, 3, 1)
        return payload

    def inverse_normalize_data(
        self,
        data: np.ndarray,
        stats: Dict[str, object],
    ) -> np.ndarray:
        """Return normalized generated signals to the original physical unit (Volts)."""
        if stats["mode"] == "trial":
            raise ValueError(
                "Trial-wise normalization cannot be uniquely inverted without "
                "the original per-trial mean/std. Use training-set normalization."
            )
        return data * np.asarray(stats["std"]) + np.asarray(stats["mean"])
