from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
from train import train_low_energy_two_model
import torch
from models import MockModel
from models import LowEnergyTwoModel
import glob
import torch.optim as optim
import argparse


def get_device(local=False):
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local:
        device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print("Using device:", device)
    return device

def load_training_data(device, local=False):
    if local:
        data_path="./train" #/Users/patrick/data/train"
    else:
        data_path="/scratch/DL24FA/train"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds

def load_data(device, local=False):
    if local:
        data_path="/Users/patrick/data"
    else:
        data_path="/scratch/DL24FA"

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


def load_model(device='cuda', local=False):
    """Load or initialize the model."""
    # model = MockModel()
    model = LowEnergyTwoModel(device=device, repr_dim=256).to(device)
    if local:
        state_dict = torch.load('best_model.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
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

    avg_losses = evaluator.evaluate_all(prober=prober, device=device)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model and save .pth file")
    parser.add_argument("--plot", action="store_true", help="display plots during training")
    parser.add_argument("--local", action="store_true", help="run on OS X")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    args = parser.parse_args()

    num_epochs = args.epochs
    train_only = args.train
    local = args.local
    plot = args.plot
    test_mode = args.test
    learning_rate = 1e-4
    repr_dim = 256


    device = get_device(local=local)
    print(f'Epochs = {num_epochs}')
    print(f'Learning rate = {learning_rate}')
    print(f'Representation dimension = {repr_dim}')
    print(f'Local execution = {local}')
    print(f'Test mode = {test_mode}')
    print(f'Training mode = {train_only}')

    if train_only:
        print('Training low energy model')
        model = LowEnergyTwoModel(device=device, repr_dim=repr_dim, training=True).to(device)
        train_loader = load_training_data(device=device, local=local) 

        predicted_states, target_states = train_low_energy_two_model(
            model=model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            test_mode=test_mode,
            plot=plot,
        )
        print()
        print('Saving low energy model in best_model.pth')
        torch.save(model.state_dict(), "best_model.pth")

    else: 
        # evaluate the model
        print('Evaluating best_model.pth')
        probe_train_ds, probe_val_ds = load_data(device, local=local)
        model = load_model(device=device, local=local)
        evaluate_model(device, model, probe_train_ds, probe_val_ds)


