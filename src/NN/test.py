from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.NN.dataset import QECDataset
from src.NN.net import VariationalAutoencoder
import numpy as np


def test_model_reconstruction_error(model: nn.Module, dataset: QECDataset,
                                    loss_func: Callable):  # no KL divergence loss when evaluating reconstruction error!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.double().to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    num_samples = len(dataset)
    with torch.no_grad():  # not necessary, but improves performance since no gradients are computed
        loss = 0
        for (batch_idx, batch) in enumerate(test_loader):
            output = model.forward(batch.to(device))
            loss += loss_func(output, batch.to(device))
    return loss / num_samples


def test_model_latent_space(model: VariationalAutoencoder, dataset: QECDataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.double().to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            z_mean, z_logvar, *_ = model.encoder.forward(batch.to(device))
    z_bar = torch.mean(z_mean)
    z_bar_var = torch.mean(torch.exp(z_logvar))
    return z_mean, z_logvar, z_bar, z_bar_var
