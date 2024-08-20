import copy

import torch
import torch.nn as nn
from torch.nn import MaxUnpool2d as MaxUnpool2d, ModuleList
import torch.nn.functional as F
from .transformer import DecoderLayer


# from src.nn.utils.max_unpool import MaxUnpool2d


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dims, distance, channels):
        super(TransformerDecoder, self).__init__()

        num_layers = 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # define structure
        self.linear = nn.Linear(latent_dims, (distance ** 2) * channels, device=device)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(distance ** 2, channels))

        self.decoders = ModuleList([copy.deepcopy(DecoderLayer(channels, 1, 100, device=device)) for _ in range(num_layers)])

        # self.decoder = DecoderLayer(channels, 1, 100)
        self.lstm = nn.LSTM(channels, 10, batch_first=True, proj_size=channels, device=device)

    def forward(self, z):
        x = self.linear(z)
        x = self.unflatten(x)
        for d in self.decoders:
            x = d(x)
        x, _ = self.lstm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims, distance, channels):
        super(Decoder, self).__init__()
        self.linear2 = nn.Linear(latent_dims, 20)
        self.dropout = nn.Dropout(0.25)
        self.bn4 = nn.BatchNorm1d(20)
        self.linear1 = nn.Linear(20, (40 * int(0.125 * distance + 3) * int(0.125 * distance + 3)))
        self.bn3 = nn.BatchNorm1d((40 * int(0.125 * distance + 3) * int(0.125 * distance + 3)))
        self.unflatten = nn.Unflatten(1, (40, int(0.125 * distance + 3), int(0.125 * distance + 3)))
        # self.max_unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        # self.max_unpool2 = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv3_2 = nn.ConvTranspose2d(40, 40, kernel_size=2, stride=1, padding=1, bias=True)
        self.deconv3_1 = nn.ConvTranspose2d(40, 20, kernel_size=2, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(20)
        # self.max_unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv2_2 = nn.ConvTranspose2d(20, 20, kernel_size=2, stride=1, padding=1, bias=True)
        self.deconv2_1 = nn.ConvTranspose2d(20, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(10)

        self.deconv1_2 = nn.ConvTranspose2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.deconv1_1 = nn.ConvTranspose2d(10, channels, kernel_size=2, stride=1, padding=1, bias=True)

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = F.relu(self.bn4(self.dropout(self.linear2(z))))
        x = F.relu(self.bn3(self.linear1(x)))
        x = self.unflatten(x)
        # x = self.max_unpool2(x, indices2, output_size=input_size2)
        x = F.interpolate(x, size=(2 * x.size()[2] - 1, 2 * x.size()[3] - 1), mode='bilinear', align_corners=True)
        x = F.relu(self.bn2(self.deconv3_1(F.relu(self.deconv3_2(x)))))
        x = F.interpolate(x, size=(2 * x.size()[2] - 1, 2 * x.size()[3] - 1), mode='bilinear', align_corners=True)
        x = F.relu(self.bn1(self.deconv2_1(F.relu(self.deconv2_2(x)))))
        x = F.interpolate(x, size=(2 * x.size()[2] - 1, 2 * x.size()[3] - 1), mode='bilinear', align_corners=True)
        x = F.tanh(self.deconv1_1(F.relu(self.deconv1_2(x))))
        return x


class DecoderDeep(nn.Module):
    def __init__(self, latent_dims: int, distance: int):
        super(DecoderDeep, self).__init__()
        self.linear4 = nn.Linear(latent_dims, 100)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear1 = nn.Linear(100, distance ** 2)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, distance, distance))

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = F.relu(self.linear4(z))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.unflatten(x)
        x = F.tanh(x)
        return x


class DecoderSkip(nn.Module):
    def __init__(self, latent_dims: int, distance: int, channels: int):
        super(DecoderSkip, self).__init__()
        self.linear2 = nn.Linear(latent_dims, 100)
        self.dropout = nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm1d(100)
        self.linear1 = nn.Linear(100 + latent_dims, (20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))))
        self.bn2 = nn.BatchNorm1d((20 * int(0.25 * (distance + 9)) * int(0.25 * (distance + 9))))
        self.unflatten = nn.Unflatten(1, (20, int(0.25 * (distance + 9)), int(0.25 * (distance + 9))))
        # self.max_unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.max_unpool2 = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(20, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.unflattenskip = nn.Unflatten(1, (latent_dims, 1, 1))
        self.deconvskip3 = nn.ConvTranspose2d(latent_dims, 10, kernel_size=int(0.5 * (distance + 3)), stride=1,
                                              padding=0,
                                              bias=False)
        self.bn1 = nn.BatchNorm2d(10)
        # self.max_unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.max_unpool1 = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(10, 10, kernel_size=2, stride=1, padding=1, bias=True)
        self.deconv1 = nn.ConvTranspose2d(10, 1, kernel_size=2, stride=1, padding=1, bias=True)
        self.deconvskip1 = nn.ConvTranspose2d(latent_dims, channels, kernel_size=distance, stride=1, padding=0,
                                              bias=False)

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = F.relu(self.bn3(self.dropout(self.linear2(z))))
        x = F.relu(self.bn2(self.linear1(torch.cat((x, z), dim=-1))))
        x = self.unflatten(x)
        x = self.max_unpool2(x, indices2, output_size=input_size2)
        x = F.relu(self.bn1(self.deconv3(x) + self.deconvskip3(self.unflattenskip(z))))
        x = self.max_unpool1(x, indices1, output_size=input_size1)
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv1(x) + self.deconvskip1(self.unflattenskip(z)))
        return x


class DecoderSimple(nn.Module):
    def __init__(self, latent_dims: int, distance: int, channels: int):
        super(DecoderSimple, self).__init__()
        self.linear2 = nn.Linear(latent_dims, 100)
        self.dropout = nn.Dropout(0.5)
        # self.linear1 = nn.Linear(100, 10 * int(0.5 * (distance + 2)) * int(0.5 * (distance + 3)))
        # self.unflatten = nn.Unflatten(1, (10, int(0.5 * (distance + 2)), int(0.5 * (distance + 3))))
        self.linear1 = nn.Linear(100, 10 * int(0.5 * (distance + 3)) * int(0.5 * (distance + 3)))
        self.unflatten = nn.Unflatten(1, (10, int(0.5 * (distance + 3)), int(0.5 * (distance + 3))))
        # self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.max_unpool = MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.deconv = nn.ConvTranspose2d(10, channels, kernel_size=2, stride=1, padding=1, bias=True)

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = F.relu(self.linear2(z))
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.unflatten(x)
        x = self.max_unpool(x, indices1, output_size=input_size1)
        x = F.tanh(self.deconv(x))  # changed from tanh to sigmoid on 29.04., undone 08.05. --> rather MSE than BCE
        return x


class DecoderDeepSkip(nn.Module):
    def __init__(self, latent_dims: int, distance: int):
        super(DecoderDeepSkip, self).__init__()
        self.linear4 = nn.Linear(latent_dims, 100)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(100 + latent_dims, 100)
        self.linear2 = nn.Linear(100 + latent_dims, 100)
        self.linear1 = nn.Linear(100 + latent_dims, distance ** 2)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, distance, distance))

    def forward(self, z, indices1, indices2, input_size1, input_size2):
        x = F.relu(self.linear4(z))
        x = self.dropout(x)
        x = F.relu(self.linear3(torch.cat((x, z), dim=1)))
        x = self.dropout(x)
        x = F.relu(self.linear2(torch.cat((x, z), dim=1)))
        x = self.dropout(x)
        x = self.linear1(torch.cat((x, z), dim=1))
        x = self.unflatten(x)
        x = F.tanh(x)
        return x


class DecoderIsing(nn.Module):
    def __init__(self, latent_dims, distance, channels):
        super(DecoderIsing, self).__init__()
        self.linear = nn.Linear(latent_dims, 128 * int((distance-1)/8 + 1) * int((distance-1)/8 + 1))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, int((distance-1)/8 + 1), int((distance-1)/8 + 1)))
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.pad = nn.CircularPad2d((distance - (8 * (distance-1)//8) + 1) // 2)

    def forward(self, z):
        x = F.relu(self.linear(z))
        x = self.unflatten(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = self.pad(x)
        return x
