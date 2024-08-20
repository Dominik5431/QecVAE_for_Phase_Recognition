import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.nn import QECDataset
from src.nn import VariationalAutoencoder, Net
from typing import Type


def test_model_reconstruction_error(model: nn.Module, dataset: QECDataset,
                                    loss_func: torch.nn.MSELoss()):  # no KL divergence loss when evaluating reconstruction error!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.double().to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    num_samples = len(dataset)
    with torch.no_grad():  # not necessary, but improves performance since no gradients are computed
        loss = 0
        for (batch_idx, batch) in enumerate(test_loader):
            output, mean, log_var = model.forward(batch[0].to(device))
            loss += loss_func(output, batch[0].to(
                device))  # default reduction='mean' for MSELoss --> reduction mean: takes average over whole batch
            loss = torch.mean(loss, dim=[1, 2, 3])
    return torch.mean(loss), torch.var(loss)


def test_model_latent_space(model: VariationalAutoencoder, dataset: Type[QECDataset]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.double().to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            if dataset.random_flip:
                z_mean, z_log_var, z, *_ = model.encoder.forward(batch[0].to(device))
            else:
                z_mean, z_log_var, z, *_ = model.encoder.forward(batch.to(device))
            flips = batch[1]
    return z_mean, z_log_var, z, flips


def test_model_predictions(model: Type[Net], dataset: Type[QECDataset]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.float().to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            z = model.forward(batch[0])
    return z
