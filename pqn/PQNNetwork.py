import torch


class PQNNetwork(torch.nn.Module):
    def __init__(self, stream_activation_function, action_dimension: int):
        super().__init__()

        self.phi = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, 4),
            torch.nn.GroupNorm(1, 32),
            stream_activation_function(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.GroupNorm(1, 64),
            stream_activation_function(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.GroupNorm(1, 64),
            stream_activation_function(),
        )

        self.fc = torch.nn.Linear(3136, action_dimension)

    def forward(self, state):
        features = self.phi(state / 255.0)
        return self.fc(features)
