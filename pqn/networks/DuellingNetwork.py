import torch
from ..modules import Stream
from ..modules import NatureDQNEncoder
from ..common import LinearEpsilon


class DuellingNetwork(torch.nn.Module):
    def __init__(self, action_dimension: int):
        super().__init__()
        self.phi = NatureDQNEncoder()
        self.value = Stream(
            input_dimension=3136, hidden_dimension=512, output_dimension=1
        )
        self.advantage = Stream(
            input_dimension=3136,
            hidden_dimension=512,
            output_dimension=action_dimension,
        )
        self.epsilon_greedy = True
        self.epsilon = LinearEpsilon()

    def get_value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features)

    def get_advantage(self, features: torch.Tensor) -> torch.Tensor:
        advantage = self.advantage(features)
        advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        return advantage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.phi(x / 255.0)
        return self.get_value(features) + self.get_advantage(features)
