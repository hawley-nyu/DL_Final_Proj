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


class VicRegJEPA(nn.Module):
    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256, repr_dim=256, training=False):
        super().__init__()
        self.encoder = Encoder(input_shape=(2, 65, 65), repr_dim=repr_dim)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=2)
        self.target_encoder = TargetEncoder(input_shape=(2, 65, 65), repr_dim=repr_dim)
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = repr_dim
        self.action_dim = 2
        self.state_dim = (2, 64, 64)
        self.output_dim = output_dim
        self.training = training

    def forward(self, states, actions):
        if self.training:
            # Copy trajectory channel over wall channel
            states[:, :, 1, :, :] = states[:, :, 0, :, :]

            target_states = self.target_encoder(states[:, 1:])  # skip first observation
            states = self.encoder(states[:, :-1])  # skip last observation

            predicted_states = [states[:, 0].clone()]
            for t in range(actions.size(1)):
                predicted_state = self.predictor(states[:, t], actions[:, t])
                predicted_states.append(predicted_state)
                if t + 1 < states.size(1):
                    states[:, t + 1] = predicted_state  # teacher forcing

            predicted_states = torch.stack(predicted_states, dim=1)
            return predicted_states, target_states

        else:
            # Copy trajectory channel over wall channel
            states[:, :, 1, :, :] = states[:, :, 0, :, :]

            predicted_state = self.encoder(states)
            predicted_states = [predicted_state.squeeze(1)]
            for t in range(actions.size(1)):
                predicted_state = self.predictor(predicted_state.squeeze(1), actions[:, t])
                predicted_states.append(predicted_state)

            predicted_states = torch.stack(predicted_states, dim=1)
            return predicted_states, None

    def loss(self, predicted_states, target_states):
        predicted_states = predicted_states[:, 1:]  # Remove initial state

        # Invariance (MSE) loss
        sim_loss = F.mse_loss(predicted_states, target_states)

        # Variance loss
        std_p = torch.sqrt(predicted_states.var(dim=0) + 1e-4)
        std_t = torch.sqrt(target_states.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_p)) + torch.mean(F.relu(1 - std_t))

        # Covariance loss
        B, T, D = target_states.size()
        pred_flat = predicted_states.view(-1, D)
        target_flat = target_states.view(-1, D)

        pred_cov = torch.cov(pred_flat.T)
        target_cov = torch.cov(target_flat.T)

        cov_loss = (pred_cov.fill_diagonal_(0).pow(2).sum() / D +
                    target_cov.fill_diagonal_(0).pow(2).sum() / D)

        return sim_loss + var_loss + 0.1 * cov_loss


class Encoder(nn.Module):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__()

        C, H, W = input_shape
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        H, W = (H - 1) // 2 + 1, (W - 1) // 2 + 1
        fc_input_dim = H * W * 64

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_input_dim, repr_dim)
        self.skip_fc = nn.Linear(C * input_shape[1] * input_shape[2], repr_dim)

    def forward(self, x):
        B, T, C, H, W = x.size()
        y = x

        # Main path
        x = x.contiguous().view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(B, T, -1)

        # Skip connection
        y = y.contiguous().view(B * T, -1)
        y = self.skip_fc(y)
        y = y.view(B, T, -1)

        return x + y


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)


class TargetEncoder(Encoder):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__(input_shape, repr_dim)