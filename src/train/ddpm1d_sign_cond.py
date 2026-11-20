import os
import argparse
import torch
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from conf.sign_cond_conf import sign_cond_conf

from models.modules.modules1D_cond import Unet1D, GaussianDiffusion1D

CFG = sign_cond_conf()
X_PATH = os.path.join(CFG.data_dir, "train", "X.npy")
Y_PATH = os.path.join(CFG.data_dir, "train", "y.npy")

# Dataset per il caricamento dei segnali EEG condizionati
class TrainingDataset(data.Dataset):
    def __init__(self, x_path, y_path, class_id):
        signals = np.load(x_path).astype(np.float32)
        labels = np.load(y_path).astype(np.int64)

        if signals.ndim != 3:
            raise ValueError(
                f"Attesa forma (N, C, L), ricevuta {signals.shape}"
            )

        if signals.shape[1] != CFG.channels:
            raise ValueError("Numero di canali EEG non corretto")

        if signals.shape[2] != CFG.sequence_length:
            raise ValueError("Lunghezza dei segnali non corretta")

        if len(signals) != len(labels):
            raise ValueError("Numero di segnali ed etichette differente")

        # Seleziona esclusivamente la classe richiesta
        mask = labels == class_id
        self.signals = signals[mask]
        self.labels = labels[mask]

        if len(self.signals) == 0:
            raise ValueError(f"Nessun campione per la classe {class_id}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Segnale EEG originale: (3, 500)
        org_data = torch.from_numpy(self.signals[idx])

        # Generazione dinamica del segnale perturbato
        cond_data = self.make_condition(org_data, snr_db=CFG.snr_db)

        return {
            'org_data': org_data,
            'cond_data': cond_data
        }

    @staticmethod
    def make_condition(signal, snr_db=20):
        """
        Crea una versione perturbata del segnale EEG
        aggiungendo rumore gaussiano.

        Args:
            signal: Tensor di forma (channels, sequence_length)
            snr_db: Signal-to-Noise Ratio in decibel

        Returns:
            Segnale perturbato, con la stessa forma dell'originale.
        """

        # Potenza RMS del segnale, calcolata separatamente per canale
        signal_rms = torch.sqrt(
            torch.mean(signal ** 2, dim=-1, keepdim=True)
        )

        # Deviazione standard del rumore in funzione dell'SNR
        noise_std = signal_rms / (10 ** (snr_db / 20))

        # Generazione del rumore gaussiano
        noise = torch.randn_like(signal) * noise_std

        # Segnale condizionale perturbato
        condition = signal + noise

        return condition

def find_latest_checkpoint(run_name, checkpoint_dir='./models'):
    """Cerca il checkpoint con lo step più avanzato per il run corrente."""
    if not os.path.exists(checkpoint_dir):
        return None

    prefix, suffix = f'{run_name}_step_', '.pt'
    checkpoints = []

    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(prefix) and filename.endswith(suffix):
            try:
                step = int(filename[len(prefix):-len(suffix)])
                checkpoints.append((step, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])

    # Restituisce il checkpoint con step maggiore
    return checkpoints[-1][1]


def load_checkpoint(checkpoint_path, model, optimizer, device, expected_class_id):
    """Carica lo stato del modello e controlla la coerenza della classe."""
    print(f"==========================================\n"
          f"Caricamento checkpoint: {checkpoint_path}\n"
          f"==========================================")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_class_id = checkpoint.get('class_id', None)
    if checkpoint_class_id is not None and checkpoint_class_id != expected_class_id:
        raise ValueError(
            f"Mismatch sulla classe: checkpoint={checkpoint_class_id}, attuale={expected_class_id}"
        )

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    global_step = checkpoint.get('global_step', 0)

    print(f"Ripresa dallo step: {global_step}\n"
          f"Checkpoint caricato con successo.\n"
          f"==========================================")
    return global_step


def save_checkpoint(model, optimizer, global_step, args):
    os.makedirs('./models', exist_ok=True)
    checkpoint_path = os.path.join('./models', f'{args.run_name}_step_{global_step}.pt')

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
        'class_id': args.class_id,
        'args': vars(args)
    }, checkpoint_path)

    print(f"Checkpoint salvato: {checkpoint_path}")


def train(args):
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        print("ATTENZIONE: CUDA non disponibile, il training girerà su CPU.")

    print(f"\n==========================================\n"
          f"Signal-Conditional Diffusion Model - EEG BCI IV 2b\n"
          f"==========================================\n"
          f"Device:          {device}\n"
          f"Class ID:        {args.class_id}\n"
          f"Batch size:      {args.batch_size}\n"
          f"Learning rate:   {args.lr}\n"
          f"Max iterations:  {args.max_iterations}\n"
          f"Save every:      {args.save_every}\n"
          f"Resume:          {args.resume}\n"
          f"==========================================\n")

    if not 0 <= args.class_id < CFG.num_classes:
        raise ValueError(f"class_id deve essere compreso tra 0 e {CFG.num_classes - 1}")

    # Inizializzazione data loader
    dataset = TrainingDataset(
        x_path=X_PATH,
        y_path=Y_PATH,
        class_id=args.class_id
    )

    sample = dataset[0]

    print("Original:", sample['org_data'].shape)
    print("Condition:", sample['cond_data'].shape)

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    if len(dataloader) == 0:
        raise ValueError("Nessun batch completo: ridurre batch_size o disattivare drop_last")
    print(f"Campioni nel dataset: {len(dataset)} | Batch per epoca: {len(dataloader)}")

    # Architettura U-Net 1D (configurazione da Tabella A4 del paper)
    model = Unet1D(
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8, 8),
        channels=args.channels,
        self_condition=True,
        resnet_block_groups=2
    ).to(device)

    # Modello di diffusione Gaussian
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=args.seq_length,
        timesteps=2000,
        sampling_timesteps=2000,
        objective='pred_noise',
        beta_schedule='linear',
        loss_type='l1',
        auto_normalize=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    global_step = 0

    # Gestione del ripristino da checkpoint
    if args.resume:
        checkpoint_path = find_latest_checkpoint(args.run_name)
        if checkpoint_path is None:
            print("\n==========================================\n"
                  "RESUME RICHIESTO: Nessun checkpoint trovato.\n"
                  "Il training partirà dallo step 0.\n"
                  "==========================================\n")
        else:
            global_step = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                device=device,
                expected_class_id=args.class_id
            )
    else:
        print("\nAvvio nuovo training da step 0. Usa --resume per riprendere un run precedente.\n")

    # Loop di addestramento
    diffusion.train()
    print(f"==========================================\n"
          f"Inizio training | Step iniziale: {global_step} / Target: {args.max_iterations}\n"
          f"==========================================")

    while global_step < args.max_iterations:
        for data_dict in dataloader:
            if global_step >= args.max_iterations:
                break

            org_data = data_dict['org_data'].to(device).float()
            cond_data = data_dict['cond_data'].to(device).float()

            loss = diffusion(org_data, cond_img=cond_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 100 == 0:
                print(f"Iterazione [{global_step}/{args.max_iterations}] - Loss: {loss.item():.6f}")

            if global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, global_step, args)

    # Salvataggio finale al termine delle iterazioni
    if global_step % args.save_every != 0:
        save_checkpoint(model, optimizer, global_step, args)

    print(f"\n==========================================\n"
          f"Training completato con successo!\n"
          f"Iterazioni totali: {global_step} | Class ID: {args.class_id}\n"
          f"==========================================")


def get_args():
    parser = argparse.ArgumentParser(description="Training 1D Conditional Diffusion Model")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size)
    parser.add_argument('--channels', type=int, default=CFG.channels)
    parser.add_argument('--lr', type=float, default=CFG.learning_rate)
    parser.add_argument('--device', type=str, default=CFG.device)
    parser.add_argument('--seq_length', type=int, default=CFG.sequence_length)
    parser.add_argument('--class_id', type=int, default=0)
    parser.add_argument('--max_iterations', type=int, default=CFG.max_iterations)
    parser.add_argument('--save_every', type=int, default=CFG.save_every)
    parser.add_argument('--resume', action='store_true', help="Riprende dall'ultimo checkpoint disponibile")
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"DDPM1D_sign_cond_EEG_class{args.class_id}"
    return args


if __name__ == '__main__':
    args = get_args()
    train(args)

'''
python -m train.ddpm1d_sign_cond --class_id 0
python -m train.ddpm1d_sign_cond --class_id 1
'''
