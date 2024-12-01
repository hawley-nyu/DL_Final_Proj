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
            states: [B, T, Ch, H, W]
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


class VicRegJEPA(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=256, repr_dim=256):
        super().__init__()
        self.repr_dim = repr_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(repr_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )

    def forward(self, states, actions):
        batch_size, seq_len = states.shape[:2]

        states_flat = states.reshape(-1, *states.shape[2:])
        encoded_states = self.encoder(states_flat)
        encoded_states = encoded_states.reshape(batch_size, seq_len, -1)
        encoded_states = encoded_states.transpose(0, 1)  # (BS, T, D) -> (T, BS, D)

        predictions = []
        current_state = encoded_states[0]

        for t in range(seq_len - 1):
            state_action = torch.cat([current_state, actions[:, t]], dim=1)
            next_state = self.predictor(state_action)
            predictions.append(next_state)
            current_state = next_state

        predictions = torch.stack(predictions, dim=0)  # (T-1, BS, D)
        return encoded_states, predictions

    def compute_loss(self, pred, target):
        # Match sequence length and batch size
        seq_len = min(pred.shape[0], target.shape[0])
        batch_size = min(pred.shape[1], target.shape[1])
        pred = pred[:seq_len, :batch_size]
        target = target[:seq_len, :batch_size]

        # Reshape and project
        pred = pred.transpose(0, 1).reshape(-1, self.repr_dim)
        target = target.transpose(0, 1).reshape(-1, self.repr_dim)

        pred = self.projector(pred)
        with torch.no_grad():
            target = self.projector(target)

        # Invariance loss
        sim_loss = F.mse_loss(pred, target)

        # Variance loss
        std_pred = torch.sqrt(pred.var(dim=0) + 0.0001)
        std_target = torch.sqrt(target.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # Covariance loss
        pred_centered = pred - pred.mean(dim=0)
        target_centered = target - target.mean(dim=0)

        pred_cov = (pred_centered.T @ pred_centered) / (pred.shape[0] - 1)
        target_cov = (target_centered.T @ target_centered) / (target.shape[0] - 1)

        pred_cov_offdiag = pred_cov - torch.diag_embed(torch.diag(pred_cov))
        target_cov_offdiag = target_cov - torch.diag_embed(torch.diag(target_cov))

        cov_loss = (pred_cov_offdiag ** 2).sum() / pred.shape[1] + \
                   (target_cov_offdiag ** 2).sum() / target.shape[1]

        # Combined loss
        loss = 10.0 * sim_loss + 10.0 * std_loss + 0.5 * cov_loss

        return loss, {
            'total_loss': loss.item(),
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item()
        }

        return loss, {
            'total_loss': loss.item(),
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item()
        }