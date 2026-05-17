import torch


class PQNBuffer:
    def __init__(
        self,
        *,
        steps_per_update: int,
        total_environments: int,
        observation_shape: tuple,
        action_dimension: int,
        observation_dtype: torch.dtype,
        device,
    ):
        self.observations = torch.empty(
            (steps_per_update, total_environments) + observation_shape,
            dtype=observation_dtype,
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

    def insert(self, step: int, observations, actions, rewards, terminations, q_values):
        self.observations[step] = observations
        self.actions[step] = actions
        self.rewards[step] = rewards
        self.terminations[step] = terminations
        self.q[step] = q_values
