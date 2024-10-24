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
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            if len(batch) == 3:
                z_mean, z_log_var, z, *_ = model.encoder.forward([batch[0].to(device), batch[1].to(device)])
                flips = batch[2]
            else:
                z_mean, z_log_var, z, *_ = model.encoder.forward(batch[0].to(device))
                flips = batch[1]
    return z_mean, z_log_var, z, flips


def test_latent_space_TraVAE(model, test_set, device):
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    mean = torch.tensor([]).to(device)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            mean = torch.cat((mean, model.partial_forward(batch.to(device))), dim=0)
    return torch.as_tensor(mean, device=device)


def test_model_predictions(model: Type[Net], dataset: Type[QECDataset]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.float().to(device)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            z = model.forward(batch[0])
    return z
