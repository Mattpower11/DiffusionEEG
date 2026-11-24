"""Valutazione dei modelli BioDiffusion adattati a EEG BCI Competition IV 2b.

Eseguire dalla cartella src: python -m test.test_eeg_dataset --model all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from conf.cls_free_conf import cls_free_conf
from conf.sign_cond_conf import sign_cond_conf
from .utility_test import Utility

# src/test/test_eeg_dataset.py -> src/ e poi DiffusionEEG/
SRC_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "left_hand", 1: "right_hand"}
CLS_CFG = cls_free_conf()
SIGN_CFG = sign_cond_conf()

# Questi parametri riproducono esattamente le inizializzazioni presenti nei
# rispettivi script di train ricevuti dall'utente (NON sono nuovi iperparametri).
CLS_DIFFUSION_STEPS = 1000
SIGN_DIFFUSION_STEPS = 2000


def _resolve_source_path(path: str | Path) -> Path:
    """Percorso configurato rispetto a src, indipendentemente dalla cwd."""
    value = Path(path).expanduser()
    return (value if value.is_absolute() else SRC_ROOT / value).resolve()


def load_eeg_test_set(data_dir: Path, channels: int, length: int, num_classes: int):
    x_path = data_dir / "test" / "X.npy"
    y_path = data_dir / "test" / "y.npy"
    for path in (x_path, y_path):
        if not path.is_file():
            raise FileNotFoundError(f"File del test set mancante: {path}")
    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    if x.ndim != 3 or x.shape[1:] != (channels, length):
        raise ValueError(f"Attesi EEG di forma (N,{channels},{length}); ottenuto {x.shape}")
    if y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"Etichette test non valide: X {x.shape}, y {y.shape}")
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError("Gli EEG del test set devono contenere valori numerici finiti")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= num_classes)):
        raise ValueError("Le etichette devono essere interi compresi nell'intervallo delle classi")
    return torch.from_numpy(x.astype(np.float32, copy=False)), torch.from_numpy(y.astype(np.int64, copy=False))


def select_class_samples(signals: torch.Tensor, labels: torch.Tensor,
                         class_id: int, count: int, seed: int):
    positions = torch.where(labels == class_id)[0].numpy()
    if len(positions) == 0:
        raise ValueError(f"Nessun campione nel test set per la classe {class_id}")
    n = len(positions) if count == 0 else min(count, len(positions))
    selected = np.random.default_rng(seed + class_id).choice(positions, size=n, replace=False)
    index = torch.as_tensor(selected, dtype=torch.long)
    return signals[index], labels[index], selected


def make_condition(real: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    """Stessa formula del TrainingDataset Signal-conditional: rumore gaussiano/RMS per canale."""
    rms = torch.sqrt(torch.mean(real ** 2, dim=-1, keepdim=True))
    noise_std = rms / (10 ** (snr_db / 20))
    rng = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(real.shape, generator=rng, dtype=real.dtype) * noise_std
    return real + noise


def find_class_free_checkpoint(explicit: Optional[Path]) -> Path:
    path = explicit if explicit is not None else (
        SRC_ROOT / "checkpoint" / "DDPM1D_CLS_FREE" / "checkpoint.pt"
    )
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint Label-conditional non trovato: {path}. "
                                "Usa --class-free-checkpoint per indicarne un altro.")
    return path


def find_signal_checkpoint(class_id: int, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint Signal-conditional non trovato: {path}")
        return path

    folder = SRC_ROOT / "models"
    pattern = re.compile(rf"DDPM1D_sign_cond_EEG_class{class_id}_step_(\d+)\.pt$")
    candidates = []
    if folder.is_dir():
        for path in folder.glob(f"DDPM1D_sign_cond_EEG_class{class_id}_step_*.pt"):
            match = pattern.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun checkpoint Signal-conditional classe {class_id} in {folder}. "
            f"Puoi specificarlo con --signal-checkpoint-{class_id}."
        )
    return max(candidates, key=lambda item: item[0])[1]


def build_cls_free(device: torch.device, checkpoint: Path, sampling_steps: int):
    from models.modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
    unet = Unet1D_cls_free(
        dim=64, dim_mults=(1, 2, 4, 8), num_classes=CLS_CFG.num_classes,
        cond_drop_prob=0.5, channels=CLS_CFG.channels
    )
    Utility.load_checkpoint(unet, checkpoint, device, allow_model_key=False,
                            allow_module_payload=True)
    diffusion = GaussianDiffusion1D_cls_free(
        unet, seq_length=CLS_CFG.sequence_length,
        timesteps=CLS_DIFFUSION_STEPS, sampling_timesteps=sampling_steps,
    ).to(device)
    diffusion.eval()
    return diffusion


def build_sign_cond(device: torch.device, checkpoint: Path, class_id: int, sampling_steps: int):
    from models.modules.modules1D_cond import Unet1D, GaussianDiffusion1D
    unet = Unet1D(
        dim=64, init_dim=None, out_dim=None,
        dim_mults=(1, 2, 4, 8, 8), channels=SIGN_CFG.channels,
        self_condition=True, resnet_block_groups=2,
    )
    Utility.load_checkpoint(unet, checkpoint, device, expected_class_id=class_id,
                            allow_model_key=True, allow_module_payload=False)
    diffusion = GaussianDiffusion1D(
        unet, seq_length=SIGN_CFG.sequence_length,
        timesteps=SIGN_DIFFUSION_STEPS, sampling_timesteps=sampling_steps,
        objective="pred_noise", beta_schedule="linear",
        loss_type="l1", auto_normalize=False,
    ).to(device)
    diffusion.eval()
    return diffusion


@torch.inference_mode()
def generate_cls_free(diffusion, labels: torch.Tensor, device: torch.device,
                      batch_size: int, cond_scale: float) -> torch.Tensor:
    generated = []
    for i in range(0, len(labels), batch_size):
        part = diffusion.sample(classes=labels[i:i + batch_size].to(device), cond_scale=cond_scale)
        # Nella versione consegnata, sample() applica unnormalize_to_zero_to_one
        # anche se il training riceveva direttamente i segnali EEG standardizzati.
        # Riporta l'output alla scala grezza prodotta dalla diffusione (non clip).
        generated.append((part * 2.0 - 1.0).cpu())
    return torch.cat(generated, dim=0)


@torch.inference_mode()
def generate_sign_cond(diffusion, condition: torch.Tensor,
                       device: torch.device, batch_size: int) -> torch.Tensor:
    generated = []
    for i in range(0, len(condition), batch_size):
        batch = condition[i:i + batch_size].to(device)
        generated.append(diffusion.sample(input_cond=batch, batch_size=len(batch)).cpu())
    return torch.cat(generated, dim=0)


def save_example_plot(real: torch.Tensor, generated: torch.Tensor,
                      path: Path, condition: Optional[torch.Tensor] = None,
                      snr_db: Optional[float] = None) -> None:
    """Una figura comparativa sui 3 canali; grafici solo illustrativi."""
    columns = [("Real", real[0]), ("Generated", generated[0])]
    if condition is not None:
        columns.insert(1, (f"Condition (SNR {snr_db:g} dB)", condition[0]))
    fig, axs = plt.subplots(3, len(columns), figsize=(5 * len(columns), 8),
                            squeeze=False, sharex=True)
    for c, (title, sample) in enumerate(columns):
        for channel, name in enumerate(("C3", "Cz", "C4")):
            axs[channel, c].plot(sample[channel].detach().cpu().numpy(), linewidth=0.9)
            axs[channel, c].set_ylabel(f"{name} (normalized)")
            if channel == 0:
                axs[channel, c].set_title(title)
            if channel == 2:
                axs[channel, c].set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_pair(model_name: str, class_id: int,
                  real: torch.Tensor, fake: torch.Tensor,
                  device: torch.device, seed: int, ds_epochs: int,
                  skip_discriminative: bool,
                  diagnostics_dir: Optional[Path] = None) -> dict:
    real_np = real.numpy().astype(np.float64, copy=False)
    fake_np = fake.numpy().astype(np.float64, copy=False)
    if fake.shape != real.shape:
        raise ValueError(f"Campioni reali/sintetici con forme differenti: {real.shape} vs {fake.shape}")
    if not torch.isfinite(fake).all():
        raise ValueError(f"Il modello {model_name} ha generato valori NaN o infiniti")
    wc = Utility.wavelet_coherence_score(real_np, fake_np)
    channel_scores = [Utility.wavelet_coherence_score(real_np, fake_np, i) for i in range(3)]
    ds = float("nan") if skip_discriminative else Utility.discriminative_score(
        real, fake, device, seed=seed + class_id,
        epochs=ds_epochs, minimum_real_samples=10,
        bounded_split=True, model_seed=seed + class_id,
        diagnostics_dir=diagnostics_dir,
    )
    return {
        "Model": model_name, "Class ID": class_id,
        "Class": CLASS_NAMES[class_id], "Samples": len(real),
        "Wavelet Coherence": wc, "Discriminative Score": ds,
        "WC C3": channel_scores[0], "WC Cz": channel_scores[1], "WC C4": channel_scores[2],
    }


def add_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = ("Wavelet Coherence", "Discriminative Score", "WC C3", "WC Cz", "WC C4")
    for _, group in frame.groupby("Model", sort=False):
        result = {"Model": group["Model"].iloc[0], "Class ID": "-",
                  "Class": "Aggregated (Mean)", "Samples": int(group["Samples"].sum())}
        for key in metric_columns:
            result[key] = float(group[key].mean(skipna=True))
        rows.append(result)
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("cls_free", "sign_cond", "all"), default="all")
    p.add_argument("--class-id", "--class_id", dest="class_id", type=int,
                   choices=(0, 1), default=None,
                   help="Se omesso, valuta entrambe le classi")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory contenente test/X.npy e test/y.npy; default: configurazione training")
    p.add_argument("--class-free-checkpoint", type=Path, default=None)
    p.add_argument("--signal-checkpoint-0", type=Path, default=None)
    p.add_argument("--signal-checkpoint-1", type=Path, default=None)
    p.add_argument("--samples-per-class", type=int, default=None,
                   help="0 = tutti i campioni disponibili")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--sampling-timesteps", type=int, default=None,
                   help="Se inferiore al training passa da DDPM a DDIM: usare solo per prove esplicite")
    p.add_argument("--cond-scale", type=float, default=None)
    p.add_argument("--snr-db", type=float, default=None)
    p.add_argument("--ds-epochs", type=int, default=None)
    p.add_argument("--ds-diagnostics", action="store_true",
                   help="Salva history.csv e summary.json delle prestazioni LSTM")
    p.add_argument("--ds-only", action="store_true",
                   help="Ricalcola solo il DS dai real.npy e generated.npy gia' esportati")
    p.add_argument("--ds-positive-control", action="store_true",
                   help="Controllo positivo: confronta real.npy con real.npy + offset (senza BioDiffusion)")
    p.add_argument("--ds-positive-offset", type=float, default=5.0,
                   help="Traslazione additiva deliberatamente evidente per il controllo positivo (default: 5.0)")
    p.add_argument("--ds-sensitivity-control", action="store_true",
                   help="Serie di controlli artificiali a offset crescenti e seed multipli; richiede --ds-only")
    p.add_argument("--ds-sensitivity-offsets", type=str, default="0,0.25,0.5,1,2,5",
                   help="Lista separata da virgole degli offset; lo zero e' controllo negativo")
    p.add_argument("--ds-sensitivity-seeds", type=int, default=5,
                   help="Numero di seed consecutivi per ogni intensita' (default: 5)")
    p.add_argument("--skip-discriminative", action="store_true",
                   help="Non addestrare la LSTM: Discriminative Score indicato come NaN")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    config = SIGN_CFG if args.model == "sign_cond" else CLS_CFG
    args.samples_per_class = config.test_samples_per_class if args.samples_per_class is None else args.samples_per_class
    args.batch_size = config.test_batch_size if args.batch_size is None else args.batch_size
    args.seed = config.test_seed if args.seed is None else args.seed
    args.ds_epochs = config.test_ds_epochs if args.ds_epochs is None else args.ds_epochs
    args.device = config.device if args.device is None else args.device
    args.snr_db = SIGN_CFG.test_snr_db if args.snr_db is None else args.snr_db
    args.cond_scale = CLS_CFG.test_cond_scale if args.cond_scale is None else args.cond_scale
    if args.output_dir is None:
        args.output_dir = _resolve_source_path(config.test_results_dir)
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.data_dir = _resolve_source_path(config.data_dir) if args.data_dir is None else args.data_dir.expanduser().resolve()
    if args.ds_only and args.skip_discriminative:
        p.error("--ds-only non e compatibile con --skip-discriminative")
    if args.ds_positive_control and not args.ds_only:
        p.error("--ds-positive-control richiede --ds-only")
    if args.ds_sensitivity_control and not args.ds_only:
        p.error("--ds-sensitivity-control richiede --ds-only")
    if args.ds_sensitivity_control and args.ds_positive_control:
        p.error("--ds-sensitivity-control e --ds-positive-control non si possono combinare")
    if args.ds_sensitivity_seeds < 1:
        p.error("--ds-sensitivity-seeds deve essere >= 1")
    try:
        args.ds_sensitivity_offsets = tuple(float(v.strip()) for v in args.ds_sensitivity_offsets.split(","))
    except ValueError:
        p.error("--ds-sensitivity-offsets deve contenere numeri separati da virgola")
    if not args.ds_sensitivity_offsets or any(not np.isfinite(v) or v < 0 for v in args.ds_sensitivity_offsets):
        p.error("--ds-sensitivity-offsets: lista non vuota di valori finiti >= 0")
    if len(set(args.ds_sensitivity_offsets)) != len(args.ds_sensitivity_offsets):
        p.error("--ds-sensitivity-offsets: non ripetere lo stesso offset")
    if args.ds_positive_control and (not np.isfinite(args.ds_positive_offset) or args.ds_positive_offset == 0):
        p.error("--ds-positive-offset deve essere finito e diverso da zero")
    if args.samples_per_class < 0 or args.batch_size < 1 or args.ds_epochs < 1:
        p.error("--samples-per-class >= 0, --batch-size >= 1, --ds-epochs >= 1")
    if not np.isfinite(args.snr_db) or not np.isfinite(args.cond_scale):
        p.error("SNR e cond-scale devono essere numeri finiti")
    if args.sampling_timesteps is not None:
        max_steps = CLS_DIFFUSION_STEPS if args.model in ("cls_free", "all") else SIGN_DIFFUSION_STEPS
        if not 1 <= args.sampling_timesteps <= max_steps:
            p.error(f"--sampling-timesteps deve essere tra 1 e {max_steps} per la modalità selezionata")
    return args


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesta ma non disponibile")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if CLS_CFG.channels != SIGN_CFG.channels or CLS_CFG.sequence_length != SIGN_CFG.sequence_length or CLS_CFG.num_classes != SIGN_CFG.num_classes:
        raise ValueError("Configurazioni Label e Signal non concordano su shape/classi EEG")
    real_all, labels_all = load_eeg_test_set(args.data_dir, CLS_CFG.channels, CLS_CFG.sequence_length, CLS_CFG.num_classes)
    class_ids = [args.class_id] if args.class_id is not None else list(range(CLS_CFG.num_classes))
    subsets = {cid: select_class_samples(real_all, labels_all, cid, args.samples_per_class, args.seed)
               for cid in class_ids}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    used_checkpoints = {}
    samplers = {}

    if args.model in ("cls_free", "all"):
        cp = find_class_free_checkpoint(args.class_free_checkpoint)
        steps = args.sampling_timesteps or CLS_CFG.test_sampling_timesteps
        if not 1 <= steps <= CLS_DIFFUSION_STEPS:
            raise ValueError("Numero di sampling timesteps Label-conditional non valido")
        print(f"Label-conditional checkpoint: {cp}")
        print(f"Label-conditional sampling: {'DDPM' if steps == CLS_DIFFUSION_STEPS else 'DDIM'}, {steps} passi")
        diffusion = build_cls_free(device, cp, steps)
        used_checkpoints['cls_free'] = str(cp)
        samplers['cls_free'] = {"method": "DDPM" if steps == CLS_DIFFUSION_STEPS else "DDIM", "steps": steps}
        for cid in class_ids:
            real, labels, indexes = subsets[cid]
            torch.manual_seed(args.seed + 100 + cid)
            if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + 100 + cid)
            fake = generate_cls_free(diffusion, labels, device, args.batch_size, args.cond_scale)
            path = output_dir / "cls_free" / CLASS_NAMES[cid]
            path.mkdir(parents=True, exist_ok=True)
            for name, arr in (("real.npy", real.numpy()), ("generated.npy", fake.numpy()),
                              ("labels.npy", labels.numpy()), ("test_indices.npy", indexes)):
                np.save(path / name, arr)
            save_example_plot(real, fake, path / "example.png")
            rows.append(evaluate_pair("cls_free", cid, real, fake, device,
                                      args.seed, args.ds_epochs, args.skip_discriminative,
                                      (output_dir / "ds_diagnostics" / "cls_free" / CLASS_NAMES[cid])
                                      if args.ds_diagnostics else None))
        del diffusion

    if args.model in ("sign_cond", "all"):
        steps = args.sampling_timesteps or SIGN_CFG.test_sampling_timesteps
        if not 1 <= steps <= SIGN_DIFFUSION_STEPS:
            raise ValueError("Numero di sampling timesteps Signal-conditional non valido")
        samplers['sign_cond'] = {"method": "DDPM" if steps == SIGN_DIFFUSION_STEPS else "DDIM", "steps": steps}
        for cid in class_ids:
            cp = find_signal_checkpoint(cid,
                                        args.signal_checkpoint_0 if cid == 0 else args.signal_checkpoint_1)
            print(f"Signal-conditional classe {cid} checkpoint: {cp}")
            print(f"Signal-conditional sampling: {'DDPM' if steps == SIGN_DIFFUSION_STEPS else 'DDIM'}, {steps} passi")
            diffusion = build_sign_cond(device, cp, cid, steps)
            used_checkpoints[f'sign_cond_class{cid}'] = str(cp)
            real, labels, indexes = subsets[cid]
            condition = make_condition(real, snr_db=args.snr_db, seed=args.seed + 200 + cid)
            torch.manual_seed(args.seed + 300 + cid)
            if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + 300 + cid)
            fake = generate_sign_cond(diffusion, condition, device, args.batch_size)
            path = output_dir / "sign_cond" / CLASS_NAMES[cid]
            path.mkdir(parents=True, exist_ok=True)
            for name, arr in (("real.npy", real.numpy()), ("condition.npy", condition.numpy()),
                              ("generated.npy", fake.numpy()), ("labels.npy", labels.numpy()),
                              ("test_indices.npy", indexes)):
                np.save(path / name, arr)
            save_example_plot(real, fake, path / "example.png", condition, args.snr_db)
            rows.append(evaluate_pair("sign_cond", cid, real, fake, device,
                                      args.seed, args.ds_epochs, args.skip_discriminative,
                                      (output_dir / "ds_diagnostics" / "sign_cond" / CLASS_NAMES[cid])
                                      if args.ds_diagnostics else None))
            del diffusion

    results = add_aggregates(pd.DataFrame(rows))
    results.to_csv(output_dir / "metrics.csv", index=False)
    metadata = {
        "dataset": str(args.data_dir / "test"),
        "models": args.model, "classes": class_ids,
        "samples_per_class_requested": args.samples_per_class,
        "seed": args.seed, "batch_size": args.batch_size,
        "snr_db": args.snr_db if args.model != 'cls_free' else None,
        "cond_scale": args.cond_scale if args.model != 'sign_cond' else None,
        "ds_epochs": None if args.skip_discriminative else args.ds_epochs,
        "checkpoints": used_checkpoints, "sampling": samplers,
        "metrics": {"wavelet_coherence": "Original repository Morlet CWT implementation; higher is better",
                    "discriminative_score": "Original repository LSTM |accuracy-0.5|; closer to 0 is better"},
        "note": "Label-conditioned samples are independent of test samples: paired WCS is descriptive, not reconstruction accuracy."
    }
    (output_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def evaluate_ds_only(args: argparse.Namespace) -> pd.DataFrame:
    """Ricalcola esclusivamente DS su campioni salvati, senza generazione/checkpoint."""
    output_dir = args.output_dir
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesta ma non disponibile")
    class_ids = [args.class_id] if args.class_id is not None else [0, 1]
    model_names = [args.model] if args.model != "all" else ["cls_free", "sign_cond"]
    rows = []
    for model_name in model_names:
        for class_id in class_ids:
            folder = output_dir / model_name / CLASS_NAMES[class_id]
            real_path, fake_path = folder / "real.npy", folder / "generated.npy"
            if not real_path.is_file():
                raise FileNotFoundError(f"EEG reali assenti: {real_path}")
            real = torch.from_numpy(np.load(real_path, allow_pickle=False).astype(np.float32))
            if args.ds_positive_control:
                # Controllo diagnostico: dati volutamente separabili, non EEG sintetici
                # BioDiffusion. L'offset e' applicato in memoria, senza sovrascrivere .npy.
                fake = real + args.ds_positive_offset
            else:
                if not fake_path.is_file():
                    raise FileNotFoundError(f"EEG sintetici assenti: {fake_path}")
                fake = torch.from_numpy(np.load(fake_path, allow_pickle=False).astype(np.float32))
            if real.shape != fake.shape or real.ndim != 3 or real.shape[1:] != (3, 500):
                raise ValueError(f"Forme EEG non compatibili in {folder}: {real.shape}, {fake.shape}")
            kind = "ds_positive_control" if args.ds_positive_control else "ds_diagnostics"
            diagnostic_folder = output_dir / kind / model_name / CLASS_NAMES[class_id]
            ds = Utility.discriminative_score(
                real, fake, device, seed=args.seed + class_id,
                epochs=args.ds_epochs, minimum_real_samples=10,
                bounded_split=True, model_seed=args.seed + class_id,
                diagnostics_dir=diagnostic_folder,
            )
            rows.append({"Model": model_name, "Class ID": class_id,
                         "Class": CLASS_NAMES[class_id], "Samples": len(real),
                         "Discriminative Score": ds})
    results = add_aggregates_ds_only(pd.DataFrame(rows))
    # Non sovrascrive metrics.csv e i segnali generati in precedenza.
    result_name = "metrics_ds_positive_control.csv" if args.ds_positive_control else "metrics_ds_diagnostics.csv"
    results.to_csv(output_dir / result_name, index=False)
    return results



def evaluate_ds_sensitivity(args: argparse.Namespace) -> pd.DataFrame:
    """Controllo di sensibilita' sulla sola LSTM: fake = real + offset.

    Riutilizza i file real.npy gia' salvati. Non usa i pesi BioDiffusion,
    non crea file fake.npy e non modifica la definizione del Discriminative Score.
    """
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesta ma non disponibile")

    class_ids = [args.class_id] if args.class_id is not None else [0, 1]
    model_names = [args.model] if args.model != "all" else ["cls_free", "sign_cond"]
    rows = []
    sensitivity_root = args.output_dir / "ds_sensitivity_control"
    sensitivity_root.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        for class_id in class_ids:
            folder = args.output_dir / model_name / CLASS_NAMES[class_id]
            real_path = folder / "real.npy"
            if not real_path.is_file():
                raise FileNotFoundError(f"EEG reali non trovati: {real_path}")
            real = torch.from_numpy(np.load(real_path, allow_pickle=False).astype(np.float32))
            if real.ndim != 3 or real.shape[1:] != (3, 500) or not torch.isfinite(real).all():
                raise ValueError(f"Dati EEG non validi in {real_path}: {tuple(real.shape)}")
            if len(real) < 10:
                raise ValueError(f"Servono almeno 10 EEG reali per classe; in {real_path}: {len(real)}")

            for offset in args.ds_sensitivity_offsets:
                # Offset applicato in memoria, senza sovrascrivere generated.npy.
                fake = real + offset
                label = f"offset_{offset:g}".replace(".", "p")
                for idx in range(args.ds_sensitivity_seeds):
                    seed = args.seed + idx
                    diagnostics_dir = sensitivity_root / model_name / CLASS_NAMES[class_id] / label / f"seed_{seed}"
                    ds = Utility.discriminative_score(
                        real, fake, device,
                        seed=seed + class_id,
                        epochs=args.ds_epochs,
                        minimum_real_samples=10,
                        bounded_split=True,
                        model_seed=seed + class_id,
                        diagnostics_dir=diagnostics_dir,
                    )
                    details_path = diagnostics_dir / "summary.json"
                    details = json.loads(details_path.read_text(encoding="utf-8"))
                    if details.get("status") != "ok":
                        raise RuntimeError(f"DS non calcolabile per {diagnostics_dir}")
                    rows.append({
                        "Model": model_name,
                        "Class ID": class_id,
                        "Class": CLASS_NAMES[class_id],
                        "Offset": offset,
                        "Seed": seed,
                        "Samples per origin": len(real),
                        "Discriminative Score": ds,
                        "Train Accuracy": details["train"]["accuracy"],
                        "Test Accuracy": details["test"]["accuracy"],
                        "Train Loss": details["train"]["loss"],
                        "Test Loss": details["test"]["loss"],
                        "Predicted Real Fraction": details["test"]["predicted_real_fraction"],
                        "Probability Std": details["test"]["probability_std"],
                        "Warnings": ";".join(details.get("warning_flags", [])),
                    })

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(args.output_dir / "metrics_ds_sensitivity_control.csv", index=False)

    summary = detail_df.groupby(["Model", "Class ID", "Class", "Offset"], sort=False).agg(
        seeds=("Seed", "count"),
        ds_mean=("Discriminative Score", "mean"),
        ds_std=("Discriminative Score", "std"),
        train_acc_mean=("Train Accuracy", "mean"),
        test_acc_mean=("Test Accuracy", "mean"),
        test_acc_std=("Test Accuracy", "std"),
        test_loss_mean=("Test Loss", "mean"),
    ).reset_index()
    summary["ds_std"] = summary["ds_std"].fillna(0.0)
    summary["test_acc_std"] = summary["test_acc_std"].fillna(0.0)
    summary.to_csv(args.output_dir / "metrics_ds_sensitivity_summary.csv", index=False)

    # Un grafico per modello per non sovrapporre popolazioni diverse.
    for model_name in model_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        model_summary = summary[summary["Model"] == model_name]
        for class_id in class_ids:
            cls = model_summary[model_summary["Class ID"] == class_id].sort_values("Offset")
            ax.errorbar(cls["Offset"], cls["ds_mean"], yerr=cls["ds_std"],
                        marker="o", capsize=3, label=CLASS_NAMES[class_id])
        ax.set_xlabel("Offset artificiale applicato all'EEG")
        ax.set_ylabel("Discriminative Score |test accuracy - 0.5|")
        ax.set_ylim(-0.02, 0.52)
        ax.set_title(f"Controllo di sensibilita' LSTM - {model_name}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(sensitivity_root / f"sensitivity_{model_name}.png", dpi=140)
        plt.close(fig)

    metadata = {
        "description": "Controllo artificiale di sensibilita' del Discriminative Score (fake=real+offset)",
        "offsets": list(args.ds_sensitivity_offsets),
        "seeds": [args.seed + i for i in range(args.ds_sensitivity_seeds)],
        "epochs": args.ds_epochs,
        "metric": "abs(test_accuracy - 0.5)",
        "note": "Il controllo dimostra sensibilita' a offset costanti, non stabilisce la qualita' degli EEG generati.",
    }
    (sensitivity_root / "evaluation.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def add_aggregates_ds_only(frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = []
    for name, group in frame.groupby("Model", sort=False):
        aggregate.append({"Model": name, "Class ID": "-", "Class": "Aggregated (Mean)",
                          "Samples": int(group["Samples"].sum()),
                          "Discriminative Score": float(group["Discriminative Score"].mean(skipna=True))})
    return pd.concat([frame, pd.DataFrame(aggregate)], ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.ds_sensitivity_control:
            results = evaluate_ds_sensitivity(args)
        else:
            results = evaluate_ds_only(args) if args.ds_only else evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Errore durante il test: {exc}", file=sys.stderr)
        return 1
    if args.ds_sensitivity_control:
        print("\nCONTROLLO DI SENSIBILITA': DS medio e deviazione standard per offset:")
    elif args.ds_positive_control:
        print("\nCONTROLLO POSITIVO: segnali deliberatamente separabili; DS atteso vicino a 0.5:")
    elif args.ds_only:
        print("\nRisultati ricalcolati (Discriminative Score; più vicino a zero):")
    else:
        print("\nRisultati (Wavelet Coherence più alta; Discriminative Score più vicino a zero):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nRisultati salvati in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
