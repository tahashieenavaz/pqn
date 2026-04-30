import torch
from .constants import PQNOptimizerType


class PQN:
    def __init__(
        self,
        return_lambda: float = 0.65,
        gamma: float = 0.99,
        optimizer: PQNOptimizerType = "radam",
    ):
        params = locals()
        params.pop("self")
        self.__initialize_hyper_parameters(params)

    def __initialize_hyper_parameters(self, params: dict):
        for key, value in params:
            setattr(self, key, value)

    def train(self, *, environment: str, seed: int):
        pass

    def log(self, *, directory: str):
        pass

    def save(self, *, directory: str):
        pass
