import torch
from typing import Type


class Stream(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            activation(inplace=True),
            torch.nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
