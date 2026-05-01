import torch
import numpy
from baloot import seed_everything
from .constants import PQNOptimizerType
from .constants import NetworkStringType
from .maps import optimizer_map
from .maps import network_map


class PQN:
    def __init__(
        self,
        network: NetworkStringType = "q",
        return_lambda: float = 0.65,
        lr: float = 0.00025,
        epochs: int = 2,
        gamma: float = 0.99,
        optimizer: PQNOptimizerType = "radam",
        optimizer_first_beta: float = 0.99,
        optimizer_second_beta: float = 0.999,
        optimizer_epsilon: float = 1e-5,
        optimizer_weight_decay: float = 0.0,
        train_environments: int = 128,
        test_environments: int = 8,
    ):
        params = locals()
        params.pop("self")
        self.__initialize_hyper_parameters(params)
        self.__initialize_derivative_parameters()
        self.__initialize_network()
        self.__initialize_optimizer()

    def __initialize_hyper_parameters(self, params: dict):
        for key, value in params:
            setattr(self, key, value)

    def __initialize_derivative_parameters(self):
        self.total_environments = self.train_environments + self.test_environments

    def __initialize_network(self, action_dimension: int):
        if self.network not in network_map:
            raise Exception(f"Optimizer `{self.optimizer}` was not founded.")

        network_instance = network_map[self.network]
        self._network = network_instance(action_dimension=action_dimension)
        self._network = torch.compile(self._network)

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
        seed_everything(seed)
        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        pass

    def log(self, *, directory: str):
        pass

    def save(self, *, directory: str):
        pass
