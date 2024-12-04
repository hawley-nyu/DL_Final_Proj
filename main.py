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

from VicRegJEPA_Train import train_vicreg
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model and save .pth file")
    parser.add_argument("--local", action="store_true", help="run on OS X")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    args = parser.parse_args()

    num_epochs = args.epochs
    train_only = args.train
    local = args.local
    test_mode = args.test
    learning_rate = 1e-4
    repr_dim = 256

    device = get_device(local=local)
    print(f'Epochs = {num_epochs}')
    print(f'Local execution = {local}')
    print(f'Learning rate = {learning_rate}')
    print(f'Representation dimension = {repr_dim}')

    if train_only:
        print('Training VicReg model')
        model = train_vicreg(device=device, repr_dim=repr_dim, training=True).to(device)
        train_loader = load_training_data(device=device, local=local)

        predicted_states, target_states = train_vicreg_model(
            model=model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            test_mode=test_mode,
        )
        print()
        print('Saving VicReg model in best_model.pth')
        torch.save(model.state_dict(), "best_model.pth")

    else:
        # evaluate the model
        print('Evaluating best_model.pth')
        probe_train_ds, probe_val_ds = load_data(device, local=local)
        model = load_model(device=device, local=local)
        evaluate_model(device, model, probe_train_ds, probe_val_ds)

'''if __name__ == "__main__":
    main()
    model = VicRegJEPA().to(device)
    train_jepa(model, train_ds)

    # Evaluate trained model
    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)'''


