import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Normalize, Compose, Resize, ToTensor
from transformers import ViTImageProcessor, ViTForImageClassification

import src.nn.net.vision_transformer


def train(model: nn.Module, init_optimizer: Callable[[Any], Optimizer], loss: Callable, epochs, batch_size,
          dataset: Dataset, val_dataset: Dataset) -> nn.Module:
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    print("Training will run on {0}".format(device))
    model = model.double().to(device)
    train_loader = DataLoader(dataset, batch_size,
                              shuffle=True)  #, pin_memory=True)  # TODO batch size and shuffle are mutually exclusive according to DataLoader docs
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=True)  #, pin_memory=True)  # TODO check if pin_memory actually makes things faster
    optimizer = init_optimizer((model.parameters()))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True) # TODO check how the scheduler works and how to optimize the hyperparameters, adjust learning rate based on loss on validation dataset, modify training and data to provide a validation dataset
    writer = SummaryWriter('logs/train')
    val_loss_increase = 0
    previous_val_loss = float('inf')
    best_val_loss = float("inf")

    dist = dataset[0][0].shape[1]

    ks = {15: 0.08,
          21: 0.09,
          27: 0.09,
          33: 0.11,
          37: 0.12,
          43: 0.13
          }

    bs = {15: 6.5,
          21: 6.5,
          27: 6.5,
          33: 6.5,
          37: 6.5,
          43: 6.5
          }

    k = ks[dist]
    b = bs[dist]

    for e in range(epochs):
        beta = (1 + np.exp(-k * e + b))
        avg_loss = 0
        num_batches = 0
        model.train()
        for (batch_idx, batch) in enumerate(train_loader):
            # print(batch_idx)
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
                val_loss = loss(val_output, val_mean, val_log_var, batch.to(device), beta=500)
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


def train_supervised(model: nn.Module, init_optimizer: Callable[[Any], Optimizer], loss: Callable, epochs, batch_size,
                     dataset: Dataset, val_dataset: Dataset) -> nn.Module:
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    print("Training will run on {0}".format(device))
    model = model.float().to(device)
    train_loader = DataLoader(dataset, batch_size,
                              shuffle=True)  # , pin_memory=True)  # TODO batch size and shuffle are mutually exclusive according to DataLoader docs
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=True)  # , pin_memory=True)  # TODO check if pin_memory actually makes things faster
    optimizer = init_optimizer((model.parameters()))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True) # TODO check how the scheduler works and how to optimize the hyperparameters, adjust learning rate based on loss on validation dataset, modify training and data to provide a validation dataset
    writer = SummaryWriter('logs/train')
    val_loss_increase = 0
    previous_val_loss = float('inf')
    best_val_loss = float("inf")

    if type(model) is src.nn.vision_transformer.VisionTransformer:
        model_name = "google/vit-base-patch16-224"
        processor = ViTImageProcessor.from_pretrained(model_name)
        # print(processor.image_mean[0])
        if dataset[0][0].shape[0] == 1:
            mu, sigma = [processor.image_mean[0]], [processor.image_std[0]]  # get default mu,sigma
        else:
            mu, sigma = processor.image_mean[:2], processor.image_std[:2]
        size = processor.size

        norm = Normalize(mean=mu, std=sigma)  # normalize image pixels range to [-1,1]

        # resize 3x32x32 to 3x224x224 -> convert to Pytorch tensor -> normalize
        _trans = Compose([
            Resize(size['height']),
            ToTensor(),
            norm
        ])

    for e in range(epochs):
        avg_loss = 0
        # accuracy = float(0)
        num_batches = 0
        model.train()
        for (batch_idx, batch) in enumerate(train_loader):
            # print(batch_idx)
            optimizer.zero_grad()
            # print(type(model))
            if type(model) is src.nn.vision_transformer.VisionTransformer:
                # print(batch[0].shape)
                # import matplotlib.pyplot as plt
                # plt.imshow(batch[0][0][0])
                # plt.show()
                # plt.imshow(_trans(batch[0])[0][0])
                # plt.show()
                # print(_trans(batch[0]).expand((batch_size, 3, size['height'], size['height'])).shape)
                output = model.forward(_trans(batch[0]).expand((batch_size, 3, size['height'], size['height'])).to(device))
            else:
                output = model.forward(batch[0].to(device))
            batch_loss = loss(output, batch[1])
            batch_loss.backward()
            avg_loss += batch_loss
            # accuracy += torch.sum(output == batch[1])  # ToDO check if proper datatype comes out of here
            optimizer.step()
            num_batches += 1
        avg_loss /= num_batches
        # accuracy /= (num_batches * batch_size)
        writer.add_scalar('training loss', avg_loss, global_step=e)
        print(f'Epoch {e + 1}/{epochs}, Loss: {avg_loss:.4f}')
        avg_val_loss = 0
        # val_accuracy = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(val_loader):
                # print(type(model))
                if type(model) is src.nn.vision_transformer.VisionTransformer:
                    val_output = model.forward(
                        _trans(batch[0]).expand((batch_size, 3, size['height'], size['height'])).to(device))
                else:
                    val_output = model.forward(batch[0].to(device))
                val_loss = loss(val_output, batch[1].to(device))
                avg_val_loss += val_loss
                # val_accuracy += torch.sum(val_output == batch[1])
                num_batches += 1
            avg_val_loss /= num_batches
            # val_accuracy /= (num_batches * batch_size)
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
