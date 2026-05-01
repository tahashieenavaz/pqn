import torch
from typing import Tuple


class LinearEpsilon:
    def __init__(self, ratio: float = 0.1, target: float = 0.001):
        self.top = 1.0
        self.target = target
        self.ratio = ratio

    def get(self, frames: int, total_frames: int) -> float:
        decay_duration = total_frames * self.ratio
        if decay_duration == 0:
            return self.top
        return max(
            self.target, self.top - (frames / decay_duration) * (self.top - self.target)
        )


class RolloutBuffer:
    def __init__(
        self,
        *,
        observation_shape: tuple,
        action_dimension: int,
        steps_per_update: int,
        total_environments: int,
        device,
    ):
        self.obs = torch.empty(
            (steps_per_update, total_environments) + observation_shape,
            dtype=torch.uint8,
            device=device,
        )
        self.act = torch.empty(
            (steps_per_update, total_environments), dtype=torch.int64, device=device
        )
        self.rew = torch.empty(
            (steps_per_update, total_environments), dtype=torch.float32, device=device
        )
        self.done = torch.empty(
            (steps_per_update, total_environments), dtype=torch.float32, device=device
        )
        self.q = torch.empty(
            (steps_per_update, total_environments, action_dimension),
            dtype=torch.float32,
            device=device,
        )

    def insert(
        self,
        step: int,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        q: torch.Tensor,
    ):
        self.obs[step] = obs
        self.act[step] = act
        self.rew[step] = rew
        self.done[step] = done
        self.q[step] = q

    def get_flattened_train_data(
        self, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation_shape = self.obs.shape[2:]
        flat_obs = self.obs[:, : self.cfg.num_train_envs].reshape(
            (-1,) + observation_shape
        )
        flat_act = self.act[:, : self.cfg.num_train_envs].reshape(-1)
        flat_tgt = targets[:, : self.cfg.num_train_envs].reshape(-1)
        return flat_obs, flat_act, flat_tgt
