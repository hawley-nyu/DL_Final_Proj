from typing import NamedTuple, Optional
import torch
import numpy as np


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device

        if probing:
            self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
            self.actions = np.load(f"{data_path}/actions.npy")
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.states = np.load(f"{data_path}/states.npy")
            self.actions = np.load(f"{data_path}/actions.npy")
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):

        if self.locations is not None:
            states = torch.from_numpy(self.states[i]).float().to(self.device)
            actions = torch.from_numpy(self.actions[i]).float().to(self.device)
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            states = torch.from_numpy(self.states[i]).float()
            actions = torch.from_numpy(self.actions[i]).float()
            locations = torch.empty(0)

        return WallSample(states=states, locations=locations, actions=actions)

def create_training_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return loader

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
