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

