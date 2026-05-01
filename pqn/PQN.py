import torch
from .constants import PQNOptimizerType
from .maps import optimizer_map


class PQN:
    def __init__(
        self,
        return_lambda: float = 0.65,
        gamma: float = 0.99,
        optimizer: PQNOptimizerType = "radam",
        optimizer_first_beta: float = 0.99,
        optimizer_second_beta: float = 0.999,
        optimizer_epsilon: float = 1e-5,
        optimizer_weight_decay: float = 0.0,
    ):
        params = locals()
        params.pop("self")
        self.__initialize_hyper_parameters(params)
        self.__initialize_network()
        self.__initialize_optimizer()

    def __initialize_hyper_parameters(self, params: dict):
        for key, value in params:
            setattr(self, key, value)

    def __initialize_network(self):
        if self.network not in network_map:
            raise Exception(f"Optimizer `{self.optimizer}` was not founded.")

    def __initialize_optimizer(self):
        if self.optimizer not in optimizer_map:
            raise Exception(f"Optimizer `{self.optimizer}` was not founded.")

        optimizer_instance = optimizer_map[self.optimizer]
        self._optimizer = optimizer_instance(
            self._network.parameters(),
            lr=self.lr,
            betas=(self.optimizer_first_beta, self.optimizer_seconds_beta),
            eps=self.optimizer_epsilon,
            weight_decay=self.optimizer_weight_decay,
        )

    def train(self, *, environment: str, seed: int):
        pass

    def log(self, *, directory: str):
        pass

    def save(self, *, directory: str):
        pass
