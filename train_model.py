from dataset import create_wall_dataloader
from my_model import JEPA
import torch
from tqdm import tqdm

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
    train_dataloader = load_data(device)
    model = JEPA().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = torch.nn.MSELoss()
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        model.train()
        for batch in train_dataloader:

            encoded_states, predicted_states = model(batch.states, batch.actions)
            loss = criterion(predicted_states, encoded_states[1:])
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
