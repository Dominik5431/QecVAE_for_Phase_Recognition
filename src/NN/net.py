import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class Net(nn.Module):
    def __init__(self, name):
        super(Net, self).__init__()
        self.name = name

    def save(self):
        torch.save(self.state_dict(), str(Path().resolve().parent) + "/data/net_{0}.pt".format(self.name))

    def load(self):
        self.load_state_dict(torch.load(str(Path().resolve().parent) + "/data/net_{0}.pt".format(self.name)))

    # Do I need to override forward method?


# TODO write tests to assure sizes work properly ad so on
class VariationalAutoencoder(Net):
    def __init__(self, latent_dims: int, distance: int, name: str):
        super(VariationalAutoencoder, self).__init__(name)
        self.encoder = VariationalEncoder(latent_dims, distance)
        self.decoder = Decoder(latent_dims, distance)

    def forward(self, x):
        z_mean, z_logvar, z, indices1, indices2, input_size1, input_size2 = self.encoder(
            x)  # add values that should be used for skip connections a return values of the encoder to give them to the decoder as additional parameters
        x = self.decoder(z, indices1, indices2, input_size1, input_size2)
        return x, z_mean, z_logvar


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, distance):
        super(VariationalEncoder, self).__init__()
        # define structure
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2, stride=1, padding=1,
                               bias=True)  # padding 'same' not possible since torch doesn't implement an equivalent operation for the transposed convolution --> use padding=1
        self.conv2 = nn.Conv2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(
            10)  # find out how to access weights of bn layer to be able to recalculate original position of latent space variable afterwards
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=2, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(20)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))),
                                100)  # got shape from size analysis: after pooling: (W-F+2P)/S + 1
        self.dropout = nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc_mean = nn.Linear((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))), latent_dims)
        self.fc_logvar = nn.Linear((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))), latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # TODO look how cuda activated and include check of gpu at start of VAE under self.device
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
        z_logvar = self.fc_logvar(
            x)  # either output log variance or use exponential activation to assure positiveness of the standard deviation
        z = z_mean + torch.exp(0.5 * z_logvar) * self.N.sample(
            z_mean.shape)  # TODO set z_mean equal 0 to avoid latent variable collapse?
        # logvar really describes the logarithm of the variance, to scale the prior with sigma with thus exponentiate and multiply times 0.5 to realize the square root
        return z_mean, z_logvar, z, indices1, indices2, input_size1, input_size2


class Decoder(nn.Module):
    def __init__(self, latent_dims, distance):
        super(Decoder, self).__init__()
        self.linear2 = nn.Linear(latent_dims, 100)
        self.dropout = nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(100)
        self.linear1 = nn.Linear(100, (20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))))
        self.bn2 = nn.BatchNorm1d((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))))
        self.unflatten = nn.Unflatten(1, (20, int(0.25 * (distance + 9)), int(0.25 * (distance + 9))))
        self.max_unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(20, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(10)
        self.max_unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.deconv1 = nn.ConvTranspose2d(10, 1, kernel_size=2, stride=1, padding=1, bias=True)

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = F.relu(self.bn3(self.dropout(self.linear2(z))))
        x = F.relu(self.bn2(self.linear1(x)))
        x = self.unflatten(x)
        x = self.max_unpool2(x, indices2, output_size=input_size2)
        x = F.relu(self.bn1(self.deconv3(x)))
        x = self.max_unpool1(x, indices1, output_size=input_size1)
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv1(x))
        return x


class Encoder_FC(nn.Module):
    def __init__(self, latent_dims: int, distance: int):
        super(Encoder_FC, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((distance - 1) * distance, 100)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(100, 100)
        self.fc_mean = nn.Linear(100, latent_dims)
        self.fc_logvar = nn.Linear(100, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(
            x)  # either output log variance or use exponential activation to assure positiveness of the standard deviation
        z = z_mean + torch.exp(0.5 * z_logvar) * self.N.sample(
            z_mean.shape)  # TODO set z_mean equal 0 to avoid latent variable collapse?
        # logvar really describes the logarithm of the variance, to scale the prior with sigma with thus exponentiate and multiply times 0.5 to realize the square root
        return z_mean, z_logvar, z, 0, 0, 0, 0


class Decoder_FC(nn.Module):
    def __init__(self, latent_dims: int, distance: int):
        super(Decoder_FC, self).__init__()
        self.linear3 = nn.Linear(latent_dims, 100)
        self.linear2 = nn.Linear(100, 100)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(100, (distance - 1) * distance)
        self.unflatten = nn.Unflatten(1, (1, distance - 1, distance))

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = self.linear3(z)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.unflatten(x)
        return x
