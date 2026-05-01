import torch
from .constants import PQNOptimizerType


class PQN:
    def __init__(
        self,
        return_lambda: float = 0.65,
        gamma: float = 0.99,
        optimizer: PQNOptimizerType = "radam",
        optimizer_first_beta: float = 0.99,
        optimizer_second_beta: float = 0.999,
        optimizer_epsilon: float = 1e-5,
    ):
        params = locals()
        params.pop("self")
        self.__initialize_hyper_parameters(params)
        self.__initialize_optimizer()

    def __initialize_hyper_parameters(self, params: dict):
        for key, value in params:
            setattr(self, key, value)

    def __initialize_optimizer(self):
        self._optimizer = optimizer_map[self.optimizer]

    def train(self, *, environment: str, seed: int):
        pass

    def log(self, *, directory: str):
        pass

    def save(self, *, directory: str):
        pass
