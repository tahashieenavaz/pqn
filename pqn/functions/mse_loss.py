import torch


def mse_loss(A: torch.Tensor, B: torch.Tensor):
    return torch.nn.functional.mse_loss(A, B)
