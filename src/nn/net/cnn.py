import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import Net


class CNN(Net):
    def __init__(self, distance: int, channels: int, name: str, cluster: bool = False):
        super(CNN, self).__init__(name, cluster=cluster)
        # define structure
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=2, stride=1, padding=1,
                               bias=True,
                               padding_mode='circular')  # padding 'same' not possible since torch doesn't implement an equivalent operation for the transposed convolution --> use padding=1
        self.conv2 = nn.Conv2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(
            10)  # find out how to access weights of bn layer to be able to recalculate original position of latent space variable afterwards
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=False)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(20)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))),
                                100)  # got shape from size analysis: after pooling: (W-F+2P)/S + 1
        self.dropout = nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 2)

    def forward(self, x):
        # calculate forward pass
        # input size
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = F.relu(self.bn3(self.dropout(self.linear(x))))
        z = F.sigmoid(self.linear2(x))
        return z
