import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from pathlib import Path


def train_jepa(
        model,
        train_loader,
        val_loader,
        probe_train_ds,
        probe_val_ds,
        num_epochs=100,
        initial_lr=1e-4,
        device="cuda",
        save_path="checkpoints",
        gradient_clip=1.0,
        validation_interval=5
):
    os.makedirs(save_path, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    wandb.init(project="jepa-training", config={
        "learning_rate": initial_lr,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "gradient_clip": gradient_clip
    })

    best_val_loss = float('inf')
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validation on {len(val_loader.dataset)} samples")

    for epoch in range(num_epochs):
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    optimizer.step()

                    batch_metrics.append(metrics)
                    total_train_loss += loss.item()

                    pbar.set_postfix({
                        'train_loss': f"{loss.item():.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })

                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation logic
        if (epoch + 1) % validation_interval == 0:
            val_metrics = validate_model(
                model, val_loader, probe_train_ds, probe_val_ds, device
            )

            metrics = {
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["val_loss"],
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            metrics.update({f"probe_{k}_loss": v for k, v in val_metrics["probe_losses"].items()})
            wandb.log(metrics)

            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    Path(save_path) / "best_model.pth"
                )

            scheduler.step(val_metrics["val_loss"])


def validate_model(model, val_loader, probe_train_ds, probe_val_ds, device):
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
    probe_losses = evaluator.evaluate_all(prober=prober)

    return {
        "val_loss": val_loss,
        "probe_losses": probe_losses
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)