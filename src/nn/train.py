import torch
import torch.nn as nn
from typing import Any, Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def train(model: nn.Module, init_optimizer: Callable[[Any], Optimizer], loss: Callable, epochs, batch_size,
          dataset: Dataset, val_dataset: Dataset, beta: float) -> nn.Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training will run on {0}".format(device))
    model = model.double().to(device)
    train_loader = DataLoader(dataset, batch_size, shuffle=True)  #, pin_memory=True)  # TODO batch size and shuffle are mutually exclusive according to DataLoader docs
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)  #, pin_memory=True)  # TODO check if pin_memory actually makes things faster
    optimizer = init_optimizer((model.parameters()))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True) # TODO check how the scheduler works and how to optimize the hyperparameters, adjust learning rate based on loss on validation dataset, modify training and data to provide a validation dataset
    writer = SummaryWriter('logs/train')
    val_loss_increase = 0
    previous_val_loss = float('inf')
    best_val_loss = float("inf")
    for e in range(epochs):
        avg_loss = 0
        num_batches = 0
        model.train()
        for (batch_idx, batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output, mean, log_var = model.forward(batch.to(device))
            # output = torch.where(output > 0.5, torch.ones_like(output[0]), torch.zeros_like(output[0]))  # included on 29.04.
            batch_loss = loss(output, mean, log_var, batch.to(device), beta)
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
                val_output, val_mean, val_log_var = model.forward(batch.to(device))
                val_loss = loss(val_output, val_mean, val_log_var, batch.to(device), beta)
                avg_val_loss += val_loss
                num_batches += 1
            avg_val_loss /= num_batches
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save()
            if avg_val_loss >= previous_val_loss:
                val_loss_increase += 1
            else:
                val_loss_increase = 0
            previous_val_loss = avg_val_loss
            writer.add_scalar('validation loss', avg_val_loss, global_step=e)
            print(f'Epoch {e + 1}/{epochs}, Validation loss: {avg_val_loss:.4f}')
            if val_loss_increase > 4:
                break
    return model

