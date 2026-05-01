import torch
import numpy
import envpool
import os
from types import SimpleNamespace
from typing import Tuple
from baloot import seed_everything
from baloot import acceleration_device
from .constants import PQNOptimizerType
from .constants import NetworkStringType
from .maps import optimizer_map
from .maps import network_map


class PQN:
    def __init__(
        self,
        network: NetworkStringType = "q",
        return_lambda: float = 0.65,
        frame_skip: int = 4,
        frames: int = 200_000_000,
        minibatches: int = 32,
        steps_per_update: int = 32,
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
        train_cpu_distribution: float = 0.9,
    ):
        torch.set_float32_matmul_precision("high")

        params = locals()
        params.pop("self")
        self.__initialize_hyper_parameters(params)
        self.__initialize_derivative_parameters()
        self.__initialize_device()

    def __initialize_hyper_parameters(self, params: dict):
        for key, value in params:
            setattr(self, key, value)

    def __initialize_derivative_parameters(self):
        self.total_environments = self.train_environments + self.test_environments

        self.total_cpu = os.cpu_count()
        self.train_cpu_count = int(self.train_cpu_distribution * self.total_cpu)
        self.test_cpu_count = self.total_cpu - self.train_cpu_count
        self.environment_steps = int(self.frames / self.frame_skip)

    def __initialize_device(self):
        self.device = acceleration_device()

    @torch.inference_mode()
    def __initialize_network(self, action_dimension: int):
        if self.network not in network_map:
            raise Exception(f"Optimizer `{self.optimizer}` was not founded.")

        network_instance = network_map[self.network]
        self._network = network_instance(action_dimension=action_dimension)
        self._network = torch.compile(self._network)
        self._network(torch.randn(1, 4, 84, 84))

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

    def __make_environments(self, environment: str, seed: int):
        train_environments = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.train_environments,
            seed=seed,
            num_threads=self.train_cpu_count,
            thread_affinity_offset=0,
            noop_max=30,
            reward_clip=True,
            episodic_life=True,
        )

        test_environments = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.test_environments,
            seed=seed + 1000,
            num_threads=self.test_cpu_count,
            thread_affinity_offset=0,
            noop_max=30,
            reward_clip=True,
            episodic_life=False,
        )

        return SimpleNamespace(
            **{
                "train": train_environments,
                "test": test_environments,
            }
        )

    def __step_environments(
        train_env, test_env, actions: numpy.ndarray, num_train: int
    ) -> Tuple[numpy.ndarray, ...]:
        train_action, test_action = actions[:num_train], actions[num_train:]

        n_obs_train, rew_train, term_train, trunc_train, info_train = train_env.step(
            train_action
        )
        n_obs_test, rew_test, term_test, trunc_test, info_test = test_env.step(
            test_action
        )

        next_observation = numpy.concatenate([n_obs_train, n_obs_test], axis=0)
        rewards = numpy.concatenate([rew_train, rew_test], axis=0)
        terms = numpy.concatenate([term_train, term_test], axis=0)
        truncs = numpy.concatenate([trunc_train, trunc_test], axis=0)
        terminations = numpy.logical_or(terms, truncs)

        infos = {}
        for k, v_train in info_train.items():
            v_test = info_test.get(k)
            if isinstance(v_train, numpy.ndarray) and isinstance(v_test, numpy.ndarray):
                infos[k] = (
                    numpy.stack([v_train, v_test])
                    if v_train.ndim == 0
                    else numpy.concatenate([v_train, v_test], axis=0)
                )

        return next_observation, rewards, terminations, infos

    def train(self, *, environment: str, seed: int):
        seed_everything(seed)
        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        overall_frame_count = 0
        environments = self.__make_environments(environment=environment, seed=seed)
        observation_shape = environments.train.observation_space.shape
        action_dimension = environments.train.action_space.n
        self.__initialize_network(action_dimension=action_dimension)
        self.__initialize_optimizer()

        train_observation, _ = environments.train.reset()
        test_observation, _ = environments.test.reset()
        observations = numpy.concatenate([train_observation, test_observation], axis=0)

        pass

    def log(self, *, directory: str):
        pass

    def save(self, *, directory: str):
        pass
