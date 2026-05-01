import torch


def mse_loss(A: torch.Tensor, B: torch.Tensor):
    return 0.5 * torch.nn.functional.mse_loss(A, B)
