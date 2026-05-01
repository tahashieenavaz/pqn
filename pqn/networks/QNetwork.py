import torch
from ..modules import Stream
from ..modules import NatureDQNEncoder
from ..common import LinearEpsilon


class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = NatureDQNEncoder()
        self.q = Stream(input_dimension=3136, hidden_dimension=512, output_dimension=1)
        self.epsilon_greedy = True
        self.epsilon = LinearEpsilon()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.phi(x / 255.0)
        return self.q(features)
