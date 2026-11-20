import os
import copy
import sys
sys.path.insert(0, './modules/')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

from models.modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
import logging
from torch.utils.tensorboard import SummaryWriter
from conf.cls_free_conf import cls_free_conf
import argparse

from torch.utils import data
from utils import setup_logging, save_signals_cls_free, save_checkpoint
from matplotlib import pyplot as plt

CFG = cls_free_conf()

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

X_PATH = os.path.join(CFG.data_dir, "train", "X.npy")
Y_PATH = os.path.join(CFG.data_dir, "train", "y.npy")

class TrainingDataset(data.Dataset):
    def __init__(self, x_path, y_path):
        self.signals = np.load(x_path).astype(np.float32)
        self.labels = np.load(y_path).astype(np.int64)

        if self.signals.ndim != 3:
            raise ValueError(
                f"Attesa forma (N, C, L), ricevuta {self.signals.shape}"
            )

        if self.signals.shape[1] != CFG.channels:
            raise ValueError(
                f"Attesi {CFG.channels} canali, "
                f"ricevuti {self.signals.shape[1]}"
            )

        if self.signals.shape[2] != CFG.sequence_length:
            raise ValueError(
                f"Attesi {CFG.sequence_length} campioni, "
                f"ricevuti {self.signals.shape[2]}"
            )

        if len(self.signals) != len(self.labels):
            raise ValueError("Numero di segnali ed etichette differente")

        if not np.isin(self.labels, np.arange(CFG.num_classes)).all():
            raise ValueError("Etichette EEG non compatibili con num_classes")

        self.signals = torch.from_numpy(self.signals)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def train(args):
    """
    Training function for the Deep Diffusion Probabilistic Model (DDPM) on 1D signals with classification.

    Parameters:
        - args (argparse.Namespace): Command-line arguments.
    """
    setup_logging(args.run_name)
    device = args.device

    dataset = TrainingDataset(x_path=X_PATH, y_path=Y_PATH)
    
    print("Numero campioni:", len(dataset))
    print("Forma segnale:", dataset[0][0].shape)
    print("Etichetta:", dataset[0][1])
    
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Define the UNet model
    model = Unet1D_cls_free(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=args.num_classes,
        cond_drop_prob=0.5,
        channels=args.channels
    ).to(device)

    # Define the Gaussian Diffusion model
    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length=args.seq_length,
        timesteps=1000
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = os.path.join("checkpoint", args.run_name, "checkpoint.pt")
    start_epoch = 0
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        logging.info(f"Resuming training from epoch {start_epoch}.")

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (signals, labels) in enumerate(pbar):
            signals = signals.to(device).to(torch.float)
            labels = labels.to(device).to(torch.long)
            
            # Calculate loss using the diffusion model
            loss = diffusion(signals, classes=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)

        # Generate and save sampled signals
        labels = torch.randint(0, args.num_classes, (10,)).to(device)
        sampled_signals = diffusion.sample(
            classes=labels,
            cond_scale=3.
        )
        
        # Save the generated signals as images
        save_signals_cls_free(sampled_signals, labels, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        plt.close("all")

        # Save model checkpoint
        is_best = False
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))

    logger.close()

def get_args():
    parser = argparse.ArgumentParser(description="Training 1D Conditional Diffusion Model")
    parser.add_argument('--run_name', type=str, default='DDPM1D_CLS_FREE')
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size)
    parser.add_argument('--channels', type=int, default=CFG.channels)
    parser.add_argument('--seq_length', type=int, default=CFG.sequence_length)
    parser.add_argument('--num_classes', type=int, default=CFG.num_classes)
    parser.add_argument('--device', type=str, default=CFG.device)
    parser.add_argument('--lr', type=float, default=CFG.learning_rate)
    parser.add_argument('--resume', action='store_true', help="Riprende dall'ultimo checkpoint disponibile")
    return parser.parse_args()

def launch():
    """
    Launch the training process with predefined parameters.
    """
    args = get_args()
    train(args)

if __name__ == '__main__':
    launch()

'''
cd src
python -m train.ddpm1d_cls_free
'''