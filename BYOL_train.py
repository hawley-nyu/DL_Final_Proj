import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import ProbingEvaluator
from models2 import BYOL


class EarlyStopping:
   def __init__(self, patience: int = 7, min_delta: float = 0):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_loss = None

   def __call__(self, val_loss: float) -> bool:
       if self.best_loss is None:
           self.best_loss = val_loss
           return False

       if val_loss < self.best_loss - self.min_delta:
           self.best_loss = val_loss
           self.counter = 0
       else:
           self.counter += 1

       return self.counter >= self.patience

def save_checkpoint(
       model: torch.nn.Module,
       optimizer: Adam,
       epoch: int,
       metrics: Dict[str, float],
       path: Path
) -> None:
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'metrics': metrics,
   }, path)
   logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(
       model: torch.nn.Module,
       optimizer: Adam,
       path: Path
) -> tuple[int, Optional[Dict[str, float]]]:
   if path.exists():
       checkpoint = torch.load(path)
       model.load_state_dict(checkpoint['model_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       return checkpoint['epoch'], checkpoint['metrics']
   return 0, None


def validate_byol(model: BYOL, probe_train_ds, probe_val_ds, device: torch.device) -> Dict[str, Any]:
    model.eval()
    evaluator = ProbingEvaluator(
        device=device,
        model=model.backbone,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )
    prober = evaluator.train_pred_prober()
    val_probe_losses = evaluator.evaluate_all(prober=prober)

    return {
        "probe_losses": val_probe_losses
    }


def train_byol(
        model: BYOL,
        train_loader: DataLoader,
        probe_train_ds,
        probe_val_ds,
        num_epochs: int = 2,
        initial_lr: float = 1e-4,
        device: str = "cuda",
        save_path: str = "checkpoints",
        gradient_clip: float = 0.5,
        patience: int = 5,
        min_delta: float = 1e-4,
        resume_from: Optional[str] = None
) -> None:
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from:
        start_epoch, _ = load_checkpoint(model, optimizer, Path(resume_from))
        logging.info(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                states = batch.states.to(device)
                actions = batch.actions.to(device)
                online_pred, target_proj = model(states, actions)
                loss = model.loss_fn(online_pred, target_proj)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
                model.update_target_network()

                total_train_loss += loss.item()
                pbar.set_postfix({
                    'train_loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        val_metrics = validate_byol(model, probe_train_ds, probe_val_ds, device)
        avg_train_loss = total_train_loss / len(train_loader)

        metrics = {
            "train_loss": avg_train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        metrics.update({f"probe_{k}_loss": v for k, v in val_metrics["probe_losses"].items()})

        if metrics["probe_normal_loss"] < best_val_loss:
            best_val_loss = metrics["probe_normal_loss"]
            save_checkpoint(model, optimizer, epoch, metrics, save_path / "best_model.pth")

        save_checkpoint(model, optimizer, epoch, metrics, save_path / f"checkpoint_epoch_{epoch + 1}.pth")

        if early_stopping(metrics["probe_normal_loss"]):
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

        scheduler.step()