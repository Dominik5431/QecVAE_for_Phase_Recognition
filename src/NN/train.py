import torch
import torch.nn as nn
from typing import Any, Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def train(model: nn.Module, init_optimizer: Callable[[Any], Optimizer], loss: Callable, epochs, batch_size,
          dataset: Dataset, val_dataset: Dataset) -> nn.Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training will run on {0}".format(device))
    model = model.double().to(device)
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    optimizer = init_optimizer((model.parameters()))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True) # TODO check how the scheduler works and how to optimize the hyperparameters, adjust learning rate based on loss on validation dataset, modify training and data to provide a validation dataset
    writer = SummaryWriter('logs/train')
    for e in range(epochs):
        avg_loss = 0
        num_batches = 0
        model.train()
        for (batch_idx, batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model.forward(batch.to(device))
            batch_loss = loss(output, batch.to(device))
            batch_loss.backward()
            avg_loss += batch_loss
            optimizer.step()
            num_batches += 1
        avg_loss /= num_batches
        writer.add_scalar('training loss', avg_loss, global_step=e)
        print(f'Epoch {e + 1}/{epochs}, Loss: {avg_loss:.4f}')
        avg_val_loss = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(val_loader):
                val_output = model.forward(batch.to(device))
                val_loss = loss(val_output, batch.to(device))
                avg_val_loss += val_loss
                num_batches += 1
            avg_val_loss /= num_batches
            writer.add_scalar('validation loss', avg_val_loss, global_step=e)
            print(f'Epoch {e + 1}/{epochs}, Validation loss: {avg_val_loss:.4f}')
    return model

