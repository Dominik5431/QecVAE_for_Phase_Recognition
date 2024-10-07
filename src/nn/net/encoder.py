import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import EncoderLayer, PositionalEncoding
import time


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, distance, channels, device: torch.device = torch.device('cpu')):
        super(VariationalEncoder, self).__init__()
        # define structure
        self.conv1_1 = nn.Conv2d(channels, 5, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.conv1_2 = nn.Conv2d(5, 5, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(5)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(5, 10, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.conv2_2 = nn.Conv2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(10)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        # self.conv3_1 = nn.Conv2d(10, 20, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        # self.conv3_2 = nn.Conv2d(20, 20, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        # self.bn3 = nn.BatchNorm2d(20)
        # self.avg_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.flatten = nn.Flatten()
        # if distance == 7:
        self.linear = nn.Linear((10 * int(0.25 * distance + 3) * int(0.25 * distance + 3)),
                                    20)  # got shape from size analysis: after pooling: (W-F+2P)/S + 1
        # else:
          #  self.linear = nn.Linear(())
        self.dropout = nn.Dropout(0.25)
        self.bn4 = nn.BatchNorm1d(20)
        self.fc_mean = nn.Linear(20, latent_dims)
        self.fc_log_var = nn.Linear(20, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.device = device



    def forward(self, x):
        # calculate forward pass
        l = x[2]
        x = x[1]

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.bn1(self.conv1_2(x)))
        input_size1 = x.size()
        x = self.avg_pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.bn2(self.conv2_2(x)))
        input_size2 = x.size()
        x = self.avg_pool2(x)
        # x = F.relu(self.conv3_1(x))
        # x = F.relu(self.bn3(self.conv3_2(x)))
        # x = self.avg_pool3(x)
        x = self.flatten(x)
        x = F.relu(self.bn4(self.dropout(self.linear(x))))
        z_mean = self.fc_mean(x)  # no tanh activation to not falsify the order parameter in latent space
        z_log_var = self.fc_log_var(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * self.N.sample(z_mean.shape).to(self.device)
        return z_mean, z_log_var, z, None, None, input_size1, input_size2


class VariationalEncoderPheno(nn.Module):
    def __init__(self, latent_dims, distance):
        super(VariationalEncoderPheno, self).__init__()

    def forward(self, x):
        pass


class VariationalEncoderSimple(nn.Module):
    def __init__(self, latent_dims: int, distance: int, channels: int, device: torch.device = torch.device('cpu')):
        super(VariationalEncoderSimple, self).__init__()
        self.conv = nn.Conv2d(channels, 10, kernel_size=2, stride=1, padding=1, padding_mode='circular',
                              bias=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(10 * int(0.5 * (distance + 2)) * int(0.5 * (distance + 3)), 100)
        self.linear = nn.Linear(10 * int(0.5 * (distance + 3)) * int(0.5 * (distance + 3)), 10)
        self.linear_log_1 = nn.Linear(4 * distance, 20)
        self.linear_log_2 = nn.Linear(20, 10)
        self.dropout = nn.Dropout(0.3)
        self.fc_mean = nn.Linear(20, latent_dims)
        self.fc_log_var = nn.Linear(20, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.device = device

    def forward(self, x):
        s = F.relu(self.conv(x[0]))
        input_size = s.size()
        s, indices = self.max_pool(s)
        s = self.flatten(s)
        s = F.relu(self.linear(s))
        s = self.dropout(s)

        l = F.relu(self.linear_log_1(x[1]))
        inf_trans = l
        l = self.dropout(l)
        l = F.relu(self.linear_log_2(l))

        z_mean = self.fc_mean(torch.concat((s, l), dim=1))
        z_log_var = self.fc_log_var(torch.concat((s, l), dim=1))
        z = z_mean + torch.exp(0.5 * z_log_var) * self.N.sample(z_mean.shape).to(self.device)
        return z_mean, z_log_var, z, indices, 0, input_size, 0, inf_trans


class VariationalEncoderDeep(nn.Module):
    def __init__(self, latent_dims: int, distance: int):
        super(VariationalEncoderDeep, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(distance ** 2, 100)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear_mean = nn.Linear(100, latent_dims)
        self.linear_log_var = nn.Linear(100, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        z_mean = self.linear_mean(x)
        z_log_var = self.linear_log_var(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * self.N.sample(
            z_mean.shape).to(self.device)
        return z_mean, z_log_var, z, None, None, None, None


class VariationalEncoderIsing(nn.Module):
    def __init__(self, latent_dims, distance, channels):
        super(VariationalEncoderIsing, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='circular')
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * int((distance - 1) / 8 + 1) * int((distance - 1) / 8 + 1), 16)
        self.linear_mean = nn.Linear(16, latent_dims)
        self.linear_log_var = nn.Linear(16, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        z_mean = self.linear_mean(x)
        z_log_var = self.linear_log_var(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * self.N.sample(
            z_mean.shape).to(self.device)
        return z_mean, z_log_var, z
