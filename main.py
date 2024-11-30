from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob

from tqdm import tqdm
from models import VicRegJEPA
from torch.utils.data import DataLoader
import wandb

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


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VicRegJEPA().to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
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


def train_jepa(
        model,
        train_loader,
        val_loader,
        probe_train_ds,
        probe_val_ds,
        num_epochs=100,
        initial_lr=1e-5,
        device="cuda",
        save_path="checkpoints"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    wandb.init(project="jepa-training")
    best_val_loss = float('inf')

    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validation on {len(val_loader.dataset)} samples")

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()

                states = batch.states.to(device)
                actions = batch.actions.to(device)

                predictions = model(states=states, actions=actions)
                loss, metrics = model.compute_loss(
                    predictions[:, 1:],
                    predictions.detach()[:, :-1]
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix({
                    'train_loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation and Probing every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()

            # Regular validation
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

            # Probing evaluation
            evaluator = ProbingEvaluator(
                device=device,
                model=model,
                probe_train_ds=probe_train_ds,
                probe_val_ds=probe_val_ds,
                quick_debug=False,
            )
            prober = evaluator.train_pred_prober()
            probe_losses = evaluator.evaluate_all(prober=prober)

            # Logging
            metrics = {
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            metrics.update({f"probe_{k}_loss": v for k, v in probe_losses.items()})
            wandb.log(metrics)

            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            for k, v in probe_losses.items():
                print(f"Probe {k} loss: {v:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': metrics,
                }, f'{save_path}/best_model.pth')

            scheduler.step(val_loss)

'''if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)'''

'''if __name__ == "__main__":
    device = get_device()

    # Train JEPA model
    train_ds = create_wall_dataloader(
        data_path="/scratch/yw7565/DL24FA/train",
        probing=False,
        device=device,
        train=True,
        batch_size=32
    )
    model = VicRegJEPA().to(device)
    train_jepa(model, train_ds)

    # Evaluate trained model
    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training data
    train_ds = create_wall_dataloader(
        data_path="/scratch/yw7565/DL24FA/train",
        probing=False,
        device=device,
        train=True,
        batch_size=8
    )

    # Validation data
    val_ds = create_wall_dataloader(
        data_path="/scratch/yw7565/DL24FA/val",
        probing=False,
        device=device,
        train=False,
        batch_size=8
    )

    # Probing datasets
    probe_train_ds = create_wall_dataloader(
        data_path="/scratch/yw7565/DL24FA/probe_normal/train",
        probing=True,
        device=device,
        train=True
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path="/scratch/yw7565/DL24FA/probe_normal/val",
        probing=True,
        device=device,
        train=False
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path="/scratch/yw7565/DL24FA/probe_wall/val",
        probing=True,
        device=device,
        train=False
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds
    }

    # Train model
    model = VicRegJEPA().to(device)
    train_jepa(
        model,
        train_ds,
        val_ds,
        probe_train_ds,
        probe_val_ds,
        device=device
    )

    # Final evaluation
    print("\nRunning final evaluation...")
    evaluate_model(device, model, probe_train_ds, probe_val_ds)