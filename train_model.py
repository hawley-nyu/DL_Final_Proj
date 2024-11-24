from dataset import create_wall_dataloader
import torch

def get_device():
    """Check for GPU availability."""
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds

if __name__ == "__main__":
    device = get_device()
    train_ds = load_data(device)
    for batch in train_ds:
        state = batch.states
        action = batch.actions
        print(state.shape)
        print(action.shape)
        break