import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(2) 
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 16 * 16, 128)

    def forward(self, x):
        """
        Input x:  (bs, 2, 65, 65)
        Output x: (bs, 32, 16, 16)
        """
        bs, channel, height, width = x.shape # (bs, 2, 65, 65)
        x = self.conv1(x) # (bs, 16, 65, 65)
        x = self.relu(x) # (bs, 16, 65, 65)
        x = self.pool1(x) # (bs, 16, 32, 32)

        x = self.conv2(x) # (bs, 32, 32, 32)
        x = self.relu(x) # (bs, 32, 32, 322)
        x = self.pool2(x) # (bs, 32, 16, 16)

        x = self.fc(x.view(bs, -1)) # (bs, 128)

        return x
    


class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(130, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
    
    def forward(self, s, u):
        """
        Input s:  (bs, 128)
        Input u:  (bs, 2)
        Output x: (bs, 128)
        """
        x = torch.cat([s, u], dim=1) # (bs, 130)
        x = self.fc1(x) # (bs, 256)
        x = self.relu(x) # (bs, 256)
        x = self.fc2(x) # (bs, 256)
        x = self.relu(x) # (bs, 256)
        x = self.fc3(x) # (bs, 128)
        return x

class JEPA(nn.Module):
    def __init__(self):
        super(JEPA, self).__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.repr_dim = 128

    def forward(self, states, actions):
        """
        Input states:  (bs, 17, 2, 65, 65)
        Input actions: (bs, 16, 2)
        Output encoded states:  (17, bs, 128)
        Output predicted states: (16, bs, 128)
        """
        bs, trajectory_length, channel, height, width = states.shape # (bs, 17, 2, 65, 65)

        reshaped_states = states.view(bs * trajectory_length, 2, 65, 65) # (bs * 17, 2, 65, 65)
        encoded_states = self.encoder(reshaped_states) # (bs * 17, 128)
        encoded_states = encoded_states.view(bs, trajectory_length, 128).permute(1, 0, 2) # (17, bs, 128)

        predicted_states = []
        for i in range(trajectory_length - 1):
            prediction = self.predictor(encoded_states[i], actions[:, i]) # (bs, 128)
            predicted_states.append(prediction)
        predicted_states = torch.stack(predicted_states, dim=0)  # (16, bs, 128)

        return encoded_states, predicted_states

