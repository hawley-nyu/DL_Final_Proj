from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob

from torch.utils.data import DataLoader
from tqdm import tqdm
from models import VicRegJEPA

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


def train_jepa(model, train_loader, num_epochs=100, lr=1e-4):
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            states = batch.states.to(device)
            actions = batch.actions.to(device)

            predictions = model(states=states, actions=actions)
            loss, metrics = model.compute_loss(
                predictions[:, 1:],
                predictions.detach()[:, :-1]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'jepa_model_epoch_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}")

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


def train_jepa(model, train_loader, num_epochs=100, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            states = batch.states.to(device)
            actions = batch.actions.to(device)

            predictions = model(states=states, actions=actions)
            loss, metrics = model.compute_loss(
                predictions[:, 1:],
                predictions.detach()[:, :-1]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'jepa_model_epoch_{epoch + 1}.pth')

'''if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)'''

if __name__ == "__main__":
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
    evaluate_model(device, model, probe_train_ds, probe_val_ds)