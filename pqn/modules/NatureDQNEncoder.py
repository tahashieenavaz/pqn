import torch
from typing import Type
from .LayerNorm2d import LayerNorm2d


class NatureDQNEncoder(torch.nn.Module):
    def __init__(self, *, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0, bias=False),
            LayerNorm2d(32),
            activation(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            LayerNorm2d(64),
            activation(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            LayerNorm2d(64),
            activation(inplace=True),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
