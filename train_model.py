from dataset import create_wall_dataloader
from my_model import JEPA
import torch
from tqdm import tqdm
import torch.nn.functional as F

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


def vicreg_loss(embeddings, target_embeddings, gamma=1, epsilon=1e-4, lambda_=1.0, mu=1.0, nu=1.0):
    """
    VICReg-inspired loss function to prevent representation collapse.

    Args:
        embeddings (torch.Tensor): Predicted embeddings, shape (T, BS, D).
        target_embeddings (torch.Tensor): Target embeddings, shape (T, BS, D).
        gamma (float): Target variance threshold.
        epsilon (float): Small constant to prevent numerical instability.
        lambda_ (float): Weight for invariance loss.
        mu (float): Weight for variance regularization.
        nu (float): Weight for covariance regularization.

    Returns:
        torch.Tensor: Total loss.
    """
    # Flatten embeddings to shape (T * BS, D) using reshape
    embeddings = embeddings.reshape(-1, embeddings.size(-1))  # (T * BS, D)
    target_embeddings = target_embeddings.reshape(-1, target_embeddings.size(-1))  # (T * BS, D)

    # Invariance Loss
    invariance_loss = F.mse_loss(embeddings, target_embeddings)

    # Variance Regularization
    batch_size, dim = embeddings.size()
    std_dev = torch.sqrt(embeddings.var(dim=0) + epsilon)
    variance_loss = torch.mean(F.relu(gamma - std_dev))

    # Covariance Regularization
    embeddings_centered = embeddings - embeddings.mean(dim=0)
    cov_matrix = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)
    cov_matrix.fill_diagonal_(0)  # Ignore diagonal elements
    covariance_loss = (cov_matrix ** 2).sum() / dim

    # Total Loss
    total_loss = lambda_ * invariance_loss + mu * variance_loss + nu * covariance_loss
    return total_loss, invariance_loss, variance_loss, covariance_loss

def new_loss(predicted_states, target_states):
    mse_loss = F.mse_loss(predicted_states, target_states)
    variance = target_states.var(dim=0).mean()
    var_loss = F.relu(1e-2 - variance).mean()
    return mse_loss + var_loss, mse_loss, var_loss

if __name__ == "__main__":
    device = get_device()
    train_dataloader = load_data(device)
    model = JEPA().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.MSELoss()
    num_epochs = 8
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        epoch_invariance_loss = 0
        epoch_variance_loss = 0
        # epoch_covariance_loss = 0
        model.train()
        for batch in train_dataloader:

            encoded_states, predicted_states = model(batch.states, batch.actions)
            loss, invariance_loss, variance_loss = new_loss(predicted_states, encoded_states[1:])
            epoch_loss += loss.item()
            epoch_invariance_loss += invariance_loss.item()
            epoch_variance_loss += variance_loss.item()
            # epoch_covariance_loss += covariance_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress_bar.update(1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Invariance Loss:{epoch_invariance_loss:.4f}, Variance Loss:{epoch_variance_loss:.4f}")
        torch.save(model.state_dict(), "best_model.pth")
    torch.save(model.state_dict(), "best_model.pth")
        
