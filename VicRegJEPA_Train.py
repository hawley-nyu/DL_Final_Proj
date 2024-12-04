import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import ProbingEvaluator
from pathlib import Path
from typing import Optional

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    print(f"Checkpoint saved to {path}")

def validate_model(model, val_loader, probe_train_ds, probe_val_ds, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            predictions, targets = model(states=states, actions=actions, training=True)
            loss = model.loss(predictions, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False
    )

    prober = evaluator.train_pred_prober()
    val_probe_losses = evaluator.evaluate_all(prober=prober)

    return {
        "val_loss": val_loss,
        "probe_losses": val_probe_losses
    }

def train_vicreg(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        probe_train_ds,
        probe_val_ds,
        num_epochs: int = 6,
        initial_lr: float = 2e-4,
        device: str = "cuda",
        save_path: str = "checkpoints",
        gradient_clip: float = 1.0,
        validation_interval: int = 1,
        early_stopping_patience: int = 10,
        resume_from: Optional[str] = None
) -> torch.nn.Module:
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    best_val_loss = float('inf')
    best_probe_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_metrics = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)

                predictions, targets = model(states=states, actions=actions, training=True)
                loss = model.loss(predictions, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()

                total_train_loss += loss.item()

                pbar.set_postfix({
                    'train_loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        avg_train_loss = total_train_loss / len(train_loader)

        if (epoch + 1) % validation_interval == 0:
            val_metrics = validate_model(model, val_loader, probe_train_ds, probe_val_ds, device)

            metrics = {
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["val_loss"],
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            metrics.update({f"probe_{k}": v for k, v in val_metrics["probe_losses"].items()})

            print(f"\nEpoch {epoch + 1}")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            probe_loss = sum(val_metrics["probe_losses"].values()) / len(val_metrics["probe_losses"])
            if probe_loss < best_probe_loss:
                best_probe_loss = probe_loss
                save_checkpoint(model, optimizer, epoch, metrics, save_path / "best_probe_model.pth")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(model, optimizer, epoch, metrics, save_path / "best_model.pth")

            if early_stopping(val_metrics["val_loss"]):
                print("Early stopping triggered")
                break

        scheduler.step()

    return model