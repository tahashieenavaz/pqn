import torch


class PQNBuffer:
    def __init__(
        self,
        *,
        steps_per_update: int,
        total_environments: int,
        observation_shape: tuple,
        device,
        action_dimension: int,
    ):
        self.observations = torch.empty(
            (steps_per_update, total_environments) + observation_shape,
            dtype=torch.uint8,
            device=device,
        )
        self.actions = torch.empty(
            (steps_per_update, total_environments), dtype=torch.int64, device=device
        )
        self.rewards = torch.empty(
            (steps_per_update, total_environments), dtype=torch.float32, device=device
        )
        self.terminations = torch.empty(
            (steps_per_update, total_environments), dtype=torch.float32, device=device
        )
        self.q = torch.empty(
            (steps_per_update, total_environments, action_dimension),
            dtype=torch.float32,
            device=device,
        )

    def insert(self, step: int, action, reward, termination, q, observation):
        self.actions[step] = action
        self.rewards[step] = reward
        self.terminations[step] = termination
        self.observation[step] = observation
        self.q[step] = q
