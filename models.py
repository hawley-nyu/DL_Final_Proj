from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim * 2)
        self.norm3 = nn.BatchNorm2d(hidden_dim * 4)
        self.fc = nn.Linear(hidden_dim * 4 * 8 * 8, output_dim)

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])

        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(B, T, -1)

        return x


class Predictor(nn.Module):
    def __init__(self, state_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def compute_covariance(x):
    batch_size = x.shape[0]
    x = x - x.mean(dim=0)
    return (x.T @ x) / (batch_size - 1)


def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VicRegJEPA(nn.Module):
    def __init__(self, sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0):
        super().__init__()
        self.encoder = ConvEncoder()  # Encθ
        self.target_encoder = ConvEncoder()  # Encψ
        self.predictor = Predictor()
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.repr_dim = 256

        # Initialize target encoder with encoder weights
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W] - full trajectory
            During inference:
                states: [B, 1, Ch, H, W] - only initial state
            actions: [B, T-1, 2]
        """
        B, T = states.shape[:2]

        if T == 1:
            # Copy trajectory channel over wall channel
            states[:, :, 1, :, :] = states[:, :, 0, :, :]

            # Inference mode - only have access to initial state
            current_state = self.encoder(states).squeeze(1)  # [B, D]
            predictions = [current_state]

            for t in range(actions.shape[1]):
                current_state = self.predictor(current_state, actions[:, t])
                predictions.append(current_state)

            predictions = torch.stack(predictions, dim=1)  # [B, T, D]
            return predictions, None

        else:
            # Training mode - have access to full trajectory

            # Copy trajectory channel over wall channel
            states[:, :, 1, :, :] = states[:, :, 0, :, :]

            # Need target representations for loss calculation
            target_states = self.target_encoder(states)  # [B, T, D]
            encoded_states = self.encoder(states)  # [B, T, D]

            # Predict next states using current state + action
            predicted_states = []
            current_state = encoded_states[:, 0]  # Start with initial encoded state

            # For each timestep, predict next state and add to predictions
            for t in range(T - 1):
                next_state = self.predictor(current_state, actions[:, t])
                predicted_states.append(next_state)
                current_state = next_state  # Use prediction as next input

            predicted_states = torch.stack(predicted_states, dim=1)  # [B, T-1, D]
            return predicted_states, target_states[:, 1:]  # Only return target states from t+1

    def compute_loss(self, predictions, targets, std_min=0.1):
        # Invariance loss
        sim_loss = F.mse_loss(predictions, targets)

        # Variance loss
        std_pred = torch.sqrt(predictions.var(dim=1) + 1e-4)
        std_target = torch.sqrt(targets.var(dim=1) + 1e-4)
        var_loss = torch.mean(F.relu(std_min - std_pred)) + torch.mean(F.relu(std_min - std_target))

        # Covariance loss
        pred_cov = compute_covariance(predictions.reshape(-1, predictions.shape[-1]))
        target_cov = compute_covariance(targets.reshape(-1, targets.shape[-1]))

        cov_loss = off_diagonal(pred_cov).pow_(2).sum() / predictions.shape[-1] + \
                   off_diagonal(target_cov).pow_(2).sum() / targets.shape[-1]

        total_loss = self.sim_loss_weight * sim_loss + \
                     self.var_loss_weight * var_loss + \
                     self.cov_loss_weight * cov_loss

        return total_loss, {
            'sim_loss': sim_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
            'total_loss': total_loss.item()
        }