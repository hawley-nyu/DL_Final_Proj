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


class LowEnergyTwoModel(nn.Module):

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
            # the input states include each step in the trajectory
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
            # the input states only include the first step in the trajectory
            predicted_state = self.encoder(states)
            predicted_states = [predicted_state.squeeze(1)]
            for t in range(actions.size(1)):
                predicted_state = self.predictor(predicted_state.squeeze(1), actions[:, t])
                predicted_states.append(predicted_state)

            predicted_states = torch.stack(predicted_states, dim=1)

            return predicted_states, None
    
    def loss(self, predicted_states, target_states):
        predicted_states = predicted_states[:, 1:]
        mse_loss = F.mse_loss(predicted_states, target_states)
        variance = target_states.var(dim=0).mean()
        var_loss = F.relu(1e-2 - variance).mean()
        B, T, repr_dim = target_states.size()
        flattened_states = target_states.view(B * T, repr_dim)  # [B*T, repr_dim]
        cov = torch.cov(flattened_states.T)
        cov_loss = (cov.fill_diagonal_(0).pow(2).sum() / cov.size(0))
        return mse_loss + var_loss #+ cov_loss


class Encoder(nn.Module):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__()

        # calculate linear layer input size
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
        x[:, :, 1, :, :] = x[:, :, 0, :, :] # copy trajectory channel over wall channel
        
        B, T, C, H, W = x.size()
        y = x
        x = x.contiguous().view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(B, T, -1) # [B, T, repr_dim]

        # skip connection
        y = y.contiguous().view(B * T, -1)  # [B * T, C * H * W]
        y = self.skip_fc(y)
        y = y.view(B, T, -1)  # Reshape back to [B, T, repr_dim]
        
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
        x = torch.cat([state, action], dim=1)
        x = self.fc(x)
        return x

class TargetEncoder(Encoder):
    def __init__(self, input_shape, repr_dim=256):
        super().__init__(input_shape, repr_dim)

