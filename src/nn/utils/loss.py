import torch


def loss_func(output, mean, log_var, target: torch.Tensor, beta=500) -> torch.Tensor:
    # reproduction_loss = torch.nn.functional.binary_cross_entropy(output, target, reduction='sum') -> try if this
    # works better
    reproduction_loss = torch.nn.MSELoss()
    # reproduction_loss = torch.nn.BCELoss()  # changed on 29.04. --> need to go to 0,1 representation for that
    kl_div_loss = 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    return beta * reproduction_loss(output, target) - kl_div_loss.mean()  # 07.05. changed dim=1 and .mean()
