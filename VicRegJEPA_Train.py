import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import ProbingEvaluator



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


def validate_model(
        model: torch.nn.Module,
        val_loader: DataLoader,
        probe_train_ds,
        probe_val_ds,
        device: torch.device
) -> Dict[str, Any]:
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            states = batch.states.to(device)
            actions = batch.actions.to(device)
            predictions = model(states=states, actions=actions)
            loss, _ = model.compute_loss(
                predictions[:, 1:],
                predictions.detach()[:, :-1]
            )
            val_loss += loss.item()

    val_loss /= len(val_loader)

    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )
    prober = evaluator.train_pred_prober()

    # Get training probe losses
    train_probe_loss = evaluator.evaluate_pred_prober(
        prober=prober,
        val_ds=probe_train_ds
    )
    logging.info(f"Train probe loss: {train_probe_loss:.4f}")

    # Get validation probe losses
    val_probe_losses = evaluator.evaluate_all(prober=prober)
    logging.info("Validation probe losses:")
    for probe_attr, loss in val_probe_losses.items():
        logging.info(f"{probe_attr} loss: {loss:.4f}")

    return {
        "val_loss": val_loss,
        "train_probe_loss": train_probe_loss,
        "probe_losses": val_probe_losses
    }
def train_jepa(
       model: torch.nn.Module,
       train_loader: DataLoader,
       val_loader: DataLoader,
       probe_train_ds,
       probe_val_ds,
       num_epochs: int = 20,
       initial_lr: float = 1e-4,
       device: str = "cuda",
       save_path: str = "checkpoints",
       gradient_clip: float = 1.0,
       validation_interval: int = 5,
       early_stopping_patience: int = 7,
       resume_from: Optional[str] = None
) -> None:
   save_path = Path(save_path)
   save_path.mkdir(exist_ok=True)

   model = model.to(device)
   optimizer = Adam(model.parameters(), lr=initial_lr)
   scheduler = lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=5, verbose=True
   )
   early_stopping = EarlyStopping(patience=early_stopping_patience)

   # Resume from checkpoint if specified
   start_epoch = 0
   if resume_from:
       start_epoch, _ = load_checkpoint(model, optimizer, Path(resume_from))
       logging.info(f"Resuming from epoch {start_epoch}")

   best_val_loss = float('inf')
   logging.info(f"Training on {len(train_loader.dataset)} samples")
   logging.info(f"Validation on {len(val_loader.dataset)} samples")

   for epoch in range(start_epoch, num_epochs):
       model.train()
       total_train_loss = 0
       batch_metrics = []

       with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
           for batch_idx, batch in enumerate(pbar):
               try:
                   optimizer.zero_grad()

                   states = batch.states.to(device)
                   actions = batch.actions.to(device)

                   predictions = model(states=states, actions=actions)
                   loss, metrics = model.compute_loss(
                       predictions[:, 1:],
                       predictions.detach()[:, :-1]
                   )

                   loss.backward()
                   torch.nn.utils.clip_grad_norm_(
                       model.parameters(),
                       max_norm=gradient_clip
                   )
                   optimizer.step()

                   batch_metrics.append(metrics)
                   total_train_loss += loss.item()

                   pbar.set_postfix({
                       'train_loss': f"{loss.item():.4f}",
                       'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                   })

               except RuntimeError as e:
                   logging.error(f"Error in batch {batch_idx}: {str(e)}")
                   continue

       avg_train_loss = total_train_loss / len(train_loader)

       # Validation phase
       if (epoch + 1) % validation_interval == 0:
           val_metrics = validate_model(
               model, val_loader, probe_train_ds, probe_val_ds, device
           )

           metrics = {
               "train_loss": avg_train_loss,
               "val_loss": val_metrics["val_loss"],
               "learning_rate": optimizer.param_groups[0]['lr']
           }
           metrics.update({
               f"probe_{k}_loss": v
               for k, v in val_metrics["probe_losses"].items()
           })

           logging.info(f"\nEpoch {epoch + 1}")
           logging.info(f"Train Loss: {avg_train_loss:.4f}")
           logging.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

           if val_metrics["val_loss"] < best_val_loss:
               best_val_loss = val_metrics["val_loss"]
               save_checkpoint(
                   model, optimizer, epoch, metrics,
                   save_path / "best_model.pth"
               )

           # Regular checkpoint
           save_checkpoint(
               model, optimizer, epoch, metrics,
               save_path / f"checkpoint_epoch_{epoch+1}.pth"
           )

           if early_stopping(val_metrics["val_loss"]):
               logging.info("Early stopping triggered")
               break

           scheduler.step(val_metrics["val_loss"])
