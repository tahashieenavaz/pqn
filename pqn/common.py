import torch
from typing import Tuple
from functools import wraps


class LinearEpsilon:
    def __init__(self, ratio: float = 0.1, target: float = 0.001):
        self.top = 1.0
        self.target = target
        self.ratio = ratio
        self.delta = self.top - self.target

    def get(self, frames: int, total_frames: int) -> float:
        decay_duration = total_frames * self.ratio

        if decay_duration <= 0:
            return self.top

        return max(
            self.target,
            self.top - (frames / decay_duration) * self.delta,
        )


def autocast():
    def decorator(callback):
        @wraps(callback)
        def wrapper(self, *args, **kwargs):
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
            ):
                return callback(self, *args, **kwargs)

        return wrapper

    return decorator
