import os
import argparse
import torch
import torch.optim as optim
import torch.utils.data as data

from modules.modules1D_cond import Unet1D, GaussianDiffusion1D
from MITBIH import mitbih_masked


# Dataset per il caricamento dei segnali ECG condizionati
class TrainingDataset(data.Dataset):
    def __init__(self, filename, class_id):
        self.cond_ECG = mitbih_masked(filename=filename, class_id=class_id)

    def __len__(self):
        return len(self.cond_ECG)

    def __getitem__(self, idx):
        data_dict = self.cond_ECG[idx]
        return {
            'org_data': data_dict['org_data'],
            'cond_data': data_dict['cond_data']
        }


def get_args():
    parser = argparse.ArgumentParser(description="Training 1D Conditional Diffusion Model")
    parser.add_argument('--run_name', type=str, default='DDPM1D_sign_cond_MITBIH_class0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--class_id', type=int, default=0)
    parser.add_argument('--max_iterations', type=int, default=1_000_000)
    parser.add_argument('--save_every', type=int, default=10_000)
    parser.add_argument('--resume', action='store_true', help="Riprende dall'ultimo checkpoint disponibile")
    return parser.parse_args()


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

    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("ATTENZIONE: CUDA non disponibile, il training girerà su CPU.")

    print(f"\n==========================================\n"
          f"Signal-Conditional Diffusion Model - MITBIH\n"
          f"==========================================\n"
          f"Device:          {device}\n"
          f"Class ID:        {args.class_id}\n"
          f"Batch size:      {args.batch_size}\n"
          f"Learning rate:   {args.lr}\n"
          f"Max iterations:  {args.max_iterations}\n"
          f"Save every:      {args.save_every}\n"
          f"Resume:          {args.resume}\n"
          f"==========================================\n")

    # Inizializzazione data loader
    data_path = "./heartbeat/mitbih_train.csv"
    dataset = TrainingDataset(filename=data_path, class_id=args.class_id)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    print(f"Campioni nel dataset: {len(dataset)} | Batch per epoca: {len(dataloader)}")

    # Architettura U-Net 1D (configurazione da Tabella A4 del paper)
    model = Unet1D(
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8, 8),
        channels=1,
        self_condition=True,
        resnet_block_groups=2
    ).to(device)

    # Modello di diffusione Gaussian
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=128,
        timesteps=2000,
        objective='pred_v',
        beta_schedule='linear',
        loss_type='l1'
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
    model.train()
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
    save_checkpoint(model, optimizer, global_step, args)

    print(f"\n==========================================\n"
          f"Training completato con successo!\n"
          f"Iterazioni totali: {global_step} | Class ID: {args.class_id}\n"
          f"==========================================")


if __name__ == '__main__':
    args = get_args()
    train(args)
