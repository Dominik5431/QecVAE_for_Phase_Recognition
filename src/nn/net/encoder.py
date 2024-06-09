import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO noise via checking of input shape
# TODO other simple models to see which one works best: try out more convolutions and only one final dense layer

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, distance, channels):
        super(VariationalEncoder, self).__init__()
        # define structure
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=2, stride=1, padding=1,
                               bias=True,
                               padding_mode='circular')  # padding 'same' not possible since torch doesn't implement an equivalent operation for the transposed convolution --> use padding=1
        self.conv2 = nn.Conv2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(
            10)  # find out how to access weights of bn layer to be able to recalculate original position of latent space variable afterwards
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=2, stride=1, padding=1, bias=True, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(20)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))),
                                100)  # got shape from size analysis: after pooling: (W-F+2P)/S + 1
        self.dropout = nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc_mean = nn.Linear(100, latent_dims)
        self.fc_logvar = nn.Linear(100, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        # calculate forward pass
        # input size
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        input_size1 = x.size()
        x, indices1 = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        input_size2 = x.size()
        x, indices2 = self.max_pool2(x)
        x = self.flatten(x)
        x = F.relu(self.bn3(self.dropout(self.linear(x))))  # Batchnorm causes activations to get very small
        z_mean = self.fc_mean(x)  # no tanh activation to not falsify the order parameter in latent space
        z_log_var = self.fc_logvar(
            x)  # either output log variance or use exponential activation to assure positiveness of the standard deviation
        z = z_mean + torch.exp(0.5 * z_log_var) * self.N.sample(
            z_mean.shape)
        # logvar really describes the logarithm of the variance, to scale the prior with sigma with thus exponentiate and multiply times 0.5 to realize the square root
        return z_mean, z_log_var, z, indices1, indices2, input_size1, input_size2


class VariationalEncoderPheno(nn.Module):
    def __init__(self, latent_dims, distance):
        super(VariationalEncoderPheno, self).__init__()

    def forward(self, x):
        pass


class VariationalEncoderSimple(nn.Module):
    def __init__(self, latent_dims: int, distance: int, channels: int):
        super(VariationalEncoderSimple, self).__init__()
        self.conv = nn.Conv2d(channels, 10, kernel_size=2, stride=1, padding=1, padding_mode='circular',
                              bias=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(10 * int(0.5 * (distance + 2)) * int(0.5 * (distance + 3)), 100)
        self.linear = nn.Linear(10 * int(0.5 * (distance + 3)) * int(0.5 * (distance + 3)), 100)
        self.dropout = nn.Dropout(0.3)
        self.fc_mean = nn.Linear(100, latent_dims)
        self.fc_logvar = nn.Linear(100, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = F.relu(self.conv(x))
        input_size = x.size()
        x, indices = self.max_pool(x)
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        x = self.dropout(x)
        z_mean = F.tanh(self.fc_mean(x))
        z_logvar = self.fc_logvar(
            x)  # either output log variance or use exponential activation to assure positiveness of the standard deviation
        z = z_mean + torch.exp(0.5 * z_logvar) * self.N.sample(
            z_mean.shape)  # TODO set z_mean equal 0 to avoid latent variable collapse?
        # logvar really describes the logarithm of the variance, to scale the prior with sigma with thus exponentiate and multiply times 0.5 to realize the square root
        return z_mean, z_logvar, z, indices, 0, input_size, 0


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
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

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
            z_mean.shape)
        return z_mean, z_log_var, z, None, None, None, None


class VariationalEncoderIsing(nn.Module):
    def __init__(self, latent_dims, distance, channels):
        super(VariationalEncoderIsing, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, padding_mode='circular')
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((64 * int((distance+7)/4) * int((distance+7)/4)), 16)
        self.linear_mean = nn.Linear(16, latent_dims)
        self.linear_log_var = nn.Linear(16, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        z_mean = self.linear_mean(x)
        z_log_var = self.linear_log_var(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * self.N.sample(
            z_mean.shape)
        return z_mean, z_log_var, z, None, None, None, None
