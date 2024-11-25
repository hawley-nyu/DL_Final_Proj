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

