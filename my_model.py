import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class OLDEncoder(nn.Module):

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
    
class OLDTargetEncoder(nn.Module):

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


class OLDPredictor(nn.Module):

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


class Encoder(nn.Module):
    def __init__(self, input_shape=(2,65,65), repr_dim=256):
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
        x[:, :, 1, :, :] = x[:, :, 0, :, :]
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
    def __init__(self, input_shape=(2,65,65), repr_dim=256):
        super().__init__(input_shape, repr_dim)


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
        Output predicted states: (17, bs, 256)
        """
        bs, trajectory_length, channel, height, width = states.shape # (bs, 17, 2, 65, 65)
        bs, action_length, action_dim = actions.shape # (bs, 16, 2)

        # states[:,:,1,:,:] = states[:,:,0,:,:]

        # encoded_states = self.encoder(states.view(bs * trajectory_length, 2, 65, 65)) # (bs * 17, 2, 65, 65)
        # encoded_states = encoded_states.view(bs, trajectory_length, 256).permute(1, 0, 2) # (17, bs, 256)
        encoded_states = self.encoder(states)
        encoded_target_states = self.target_encoder(states)
        # encoded_target_states = self.target_encoder(states.view(bs * trajectory_length, 2, 65, 65)) 
        # encoded_target_states = encoded_target_states.view(bs, trajectory_length, 256).permute(1, 0, 2) # (17, bs, 256)

        predicted_states = []
        predicted_states.append(encoded_states[:,0])
        for i in range(action_length):
            prediction = self.predictor(predicted_states[i], actions[:,i]) # (bs, 256)
            predicted_states.append(prediction)
        predicted_states = torch.stack(predicted_states, dim=1)  # (16, bs, 256)

        return encoded_target_states, predicted_states

