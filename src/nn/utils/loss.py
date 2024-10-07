import torch
import torch.nn.functional as F


def loss_func(output, mean, log_var, z, target: torch.Tensor, beta, lambda_l1=1e-3) -> torch.Tensor:
    # reproduction_loss = torch.nn.functional.binary_cross_entropy(output, target, reduction='sum') -> try if this
    # works better
    # reconstruction_loss = torch.nn.MSELoss()
    # reproduction_loss = torch.nn.BCELoss()  # changed on 29.04. --> need to go to 0,1 representation for that
    # kl_div_loss = 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

    # Append start token also to target tensor
    device = output.device

    # start_token_value = 1
    # start_token = torch.full((target.size(0), 1), start_token_value, dtype=torch.long, device=device)
    # target = torch.cat((start_token, target), dim=1)

    # print(output.size())
    # print(target.size())

    # Reconstruction loss (Cross-Entropy)
    # reconstruction_loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), reduction='sum')
    reconstruction_loss_func = torch.nn.BCELoss(reduction='sum')
    output = output.squeeze()
    target = target.float()
    reconstruction_loss = reconstruction_loss_func(output, target)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # L1 regularization on latent space
    # l1 = lambda_l1 * torch.sum(torch.abs(z))

    return beta * reconstruction_loss + kl_loss  # + l1
    # return beta * (reconstruction_loss(output[0], target[0]) + reconstruction_loss(output[1], target[1])) - kl_div_loss.mean()  # 07.05. changed dim=1 and .mean()


def loss_func_MSE(output, mean, log_var, target: torch.Tensor, beta, lambda_l1=1e-3) -> torch.Tensor:
    # reproduction_loss = torch.nn.functional.binary_cross_entropy(output, target, reduction='sum') -> try if this
    # works better
    reconstruction_loss = torch.nn.MSELoss()
    # reproduction_loss = torch.nn.BCELoss()  # changed on 29.04. --> need to go to 0,1 representation for that
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # Append start token also to target tensor
    device = output.device

    # start_token_value = 1
    # start_token = torch.full((target.size(0), 1), start_token_value, dtype=torch.long, device=device)
    # target = torch.cat((start_token, target), dim=1)

    # print(output.size())
    # print(target.size())

    # Reconstruction loss (Cross-Entropy)
    # reconstruction_loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), reduction='sum')
    # reconstruction_loss_func = torch.nn.BCELoss(reduction='sum')
    # output = output.squeeze()
    # target = target.float()
    # reconstruction_loss = reconstruction_loss_func(output, target)

    # KL divergence loss
    # kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # L1 regularization on latent space
    # l1 = lambda_l1 * torch.sum(torch.abs(z))
    if type(output) is list:
        return beta * (reconstruction_loss(output[0], target[0]) + reconstruction_loss(output[1], target[1])) - kl_loss.mean()
    else:
        return beta * reconstruction_loss(output, target) + kl_loss  # + l1