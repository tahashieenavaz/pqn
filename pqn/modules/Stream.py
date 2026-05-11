import torch
from typing import Type


class Stream(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        normalization: bool,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        self.normalization = normalization
        self.first_linear = torch.nn.Linear(input_dimension, hidden_dimension)
        self.second_linear = torch.nn.Linear(hidden_dimension, output_dimension)
        self.activation = activation()
        if normalization:
            self.normalization_layer = torch.nn.LayerNorm(hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear(x)
        if self.normalization:
            x = self.normalization_layer(x)
        x = self.activation(x)
        x = self.second_linear(x)
        return x
