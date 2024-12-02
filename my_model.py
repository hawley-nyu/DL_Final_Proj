import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(16 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        

    def forward(self, x):
        """
        Input x:  (bs, 2, 65, 65)
        Output x: (bs, 32, 16, 16)
        """
        # x[:, 1, :, :] = x[:, 0, :, :]
        bs, channel, height, width = x.shape # (bs, 2, 65, 65)
        x = self.conv_block1(x) # (bs, 32, 32, 32)
        # identity = x
        # x = self.conv_block2(x) # (bs, 64, 32, 32)
        # x = self.conv_block3(x) # (bs, 64, 32, 32)
        # x = self.conv_block4(x) # (bs, 32, 32, 32)
        # x = x + identity # (bs, 32, 32, 32)
        x = self.conv_block5(x) # (bs, 16, 16, 16)
        x = self.linear_block(x.view(bs, -1)) # (bs, 256)
        return x
    
class TargetEncoder(nn.Module):

    def __init__(self):
        super(TargetEncoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(16 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        

    def forward(self, x):
        """
        Input x:  (bs, 2, 65, 65)
        Output x: (bs, 32, 16, 16)
        """
        # x[:, 1, :, :] = x[:, 0, :, :]
        bs, channel, height, width = x.shape # (bs, 2, 65, 65)
        x = self.conv_block1(x) # (bs, 32, 32, 32)
        # identity = x
        # x = self.conv_block2(x) # (bs, 64, 32, 32)
        # x = self.conv_block3(x) # (bs, 64, 32, 32)
        # x = self.conv_block4(x) # (bs, 32, 32, 32)
        # x = x + identity # (bs, 32, 32, 32)
        x = self.conv_block5(x) # (bs, 16, 16, 16)
        x = self.linear_block(x.view(bs, -1)) # (bs, 256)
        return x


class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()

        self.action_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        self.linear_block = nn.Sequential(
            nn.Linear(256 + 32, 512),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, state, action):
        """
        Input s:  (bs, 256)
        Input u:  (bs, 2)
        Output x: (bs, 256)
        """
        action = self.action_embedding(action) # (bs, 16)
        x = torch.cat([state, action], dim=1) # (bs, 256 + 16)
        x = self.linear_block(x) # (bs, 256)
        return x

class JEPA(nn.Module):
    def __init__(self):
        super(JEPA, self).__init__()
        self.encoder = Encoder()
        self.target_encoder = TargetEncoder()
        self.predictor = Predictor()
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Input states:  (bs, 17, 2, 65, 65)
        Input actions: (bs, 16, 2)
        Output encoded states:  (17, bs, 256)
        Output predicted states: (16, bs, 256)
        """
        bs, trajectory_length, channel, height, width = states.shape # (bs, 17, 2, 65, 65)

        # reshaped_states = states.view(bs * trajectory_length, 2, 65, 65) # (bs * 17, 2, 65, 65)
        states[:,:,1,:,:] = states[:,:,0,:,:]
        encoded_states = self.encoder(states[:,:-1,:,:,:])
        encoded_target_states = self.target_encoder(states[:, 1:, :, :, :])
        # encoded_states = encoded_states.view(bs, trajectory_length, 256).permute(1, 0, 2) # (17, bs, 256)

        predicted_states = []
        predicted_states.append(encoded_states[0])
        for i in range(trajectory_length - 1):
            prediction = self.predictor(encoded_target_states[i], actions[:, i]) # (bs, 256)
            predicted_states.append(prediction)
        predicted_states = torch.stack(predicted_states, dim=0)  # (16, bs, 256)

        return encoded_states, predicted_states

