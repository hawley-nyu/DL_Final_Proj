from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    layers.append(nn.Flatten())
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


class BYOL(torch.nn.Module):
    def __init__(self, backbone, projection_dim=256, hidden_dim=4096):
        super().__init__()
        self.backbone = backbone
        self.projector = build_mlp([backbone.repr_dim, hidden_dim, projection_dim])
        self.predictor = build_mlp([projection_dim, hidden_dim, projection_dim])

        backbone_kwargs = {
            'device': backbone.device,
            'bs': backbone.bs,
            'n_steps': backbone.n_steps,
            'img_size': 64,
            'patch_size': 8,
            'in_channels': 2,
            'embed_dim': backbone.repr_dim,
            'num_heads': 8,
            'num_layers': 6,
            'mlp_ratio': 4
        }

        self.target_backbone = type(backbone)(**backbone_kwargs)
        self.target_projector = build_mlp([backbone.repr_dim, hidden_dim, projection_dim])

        self.update_target_network(tau=1.0)
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, states, actions):
        online_init, online_preds = self.backbone(states, actions)
        online_repr = torch.cat([online_init, online_preds], dim=0)
        B, T, D = online_repr.shape
        online_repr = online_repr.reshape(B * T, D)

        online_proj = self.projector(online_repr)
        online_pred = self.predictor(online_proj)

        with torch.no_grad():
            target_init, target_preds = self.target_backbone(states, actions)
            target_repr = torch.cat([target_init, target_preds], dim=0)
            target_repr = target_repr.reshape(B * T, D)
            target_proj = self.target_projector(target_repr)

        return online_pred, target_proj

    def update_target_network(self, tau=0.996):
        """Update target network using exponential moving average"""
        for online, target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def loss_fn(self, online_pred, target_proj):
        """BYOL loss: negative cosine similarity"""
        online_pred = F.normalize(online_pred, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1)
        return 2 - 2 * (online_pred * target_proj).sum(dim=-1).mean()