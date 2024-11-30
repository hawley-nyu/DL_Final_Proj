import os
import logging
from pathlib import Path
from typing import Tuple

from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob

from tqdm import tqdm
from models import VicRegJEPA
from torch.utils.data import DataLoader, random_split

from VicRegJEPA_Train import train_jepa
from dataclasses import dataclass

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


'''def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VicRegJEPA().to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    return model'''

def load_model(checkpoint_path=None):
   device = get_device()
   model = VicRegJEPA().to(device)
   if checkpoint_path:
       model.load_state_dict(torch.load(checkpoint_path))
       logging.info(f"Loaded checkpoint from {checkpoint_path}")
   model.eval()
   return model

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

'''if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)'''


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

    device = get_device()

    full_dataset = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
        batch_size=32
    )

    train_size = int(0.9 * len(full_dataset.dataset))
    val_size = len(full_dataset.dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset.dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2000)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = VicRegJEPA().to(device)
    probe_train_ds, probe_val_ds = load_data(device)

    train_jepa(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        num_epochs=100,
        initial_lr=1e-4,
        device=device,
        save_path="checkpoints"
    )
    evaluate_model(device, model, probe_train_ds, probe_val_ds)


if __name__ == "__main__":
    main()

'''if __name__ == "__main__":
    main()
    model = VicRegJEPA().to(device)
    train_jepa(model, train_ds)

    # Evaluate trained model
    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)'''


