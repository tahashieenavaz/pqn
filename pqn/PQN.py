import torch
import numpy
import envpool
import os
import time
from pathlib import Path
from types import SimpleNamespace
from baloot import funnel, seed_everything, acceleration_device
from pqn.common import LinearEpsilon, autocast
from pqn.constants import PQNOptimizerType, NetworkStringType
from pqn.functions import (
    epsilon_greedy_vectorized,
    mse_loss,
    lambda_returns,
    to_float_list,
)
from pqn.maps import optimizer_map, network_map
from pqn.buffers import PQNBuffer


class PQN:
    def __init__(
        self,
        network: NetworkStringType = "q",
        return_lambda: float = 0.65,
        frames: int = 200_000_000,
        frame_skip: int = 4,
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
        epsilon_greedy: bool = True,
        noop: int = 30,
    ):
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

        torch.set_float32_matmul_precision("high")
        self.device = acceleration_device()
        self._epsilon = LinearEpsilon()

        self.total_environments = self.train_environments + self.test_environments
        self.total_cpu = os.cpu_count() or 1
        self.train_cpu_count = int(self.train_cpu_distribution * self.total_cpu)
        self.test_cpu_count = self.total_cpu - self.train_cpu_count
        self.effective_frames = int(self.frames / self.frame_skip)
        self.total_updates = int(
            self.effective_frames / self.train_environments / self.steps_per_update
        )
        if (self.train_environments * self.steps_per_update) % self.minibatches != 0:
            raise ValueError(
                "minibatches must divide train_environments * steps_per_update"
            )

    @torch.inference_mode()
    def __initialize_network(self, action_dimension: int):
        self._network = network_map[self.network](action_dimension=action_dimension).to(
            self.device
        )
        self._network(torch.randn(1, 4, 84, 84, device=self.device))
        self._network = torch.compile(self._network)

    def __initialize_optimizer(self):
        self._optimizer = optimizer_map[self.optimizer](
            self._network.parameters(),
            lr=self.lr,
            eps=self.optimizer_epsilon,
            betas=(self.optimizer_first_beta, self.optimizer_second_beta),
            weight_decay=self.optimizer_weight_decay,
        )

    def __environment_factory(
        self, environment: str, environments: int, cpu_count: int, life: bool, seed: int
    ):
        return envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=environments,
            seed=seed,
            num_threads=cpu_count,
            thread_affinity_offset=0,
            noop_max=self.noop,
            frame_skip=self.frame_skip,
            repeat_action_probability=0.0,
            reward_clip=True,
            episodic_life=life,
        )

    def __make_environments(self, environment: str, seed: int):
        return SimpleNamespace(
            train=self.__environment_factory(
                environment=environment,
                environments=self.train_environments,
                cpu_count=self.train_cpu_count,
                life=True,
                seed=seed,
            ),
            test=self.__environment_factory(
                environment=environment,
                environments=self.test_environments,
                cpu_count=self.test_cpu_count,
                life=False,
                seed=seed + 1000,
            ),
        )

    def train(self, *, environment: str, seed: int):
        started_at = time.perf_counter()
        results = SimpleNamespace(loss=[], test=[], train=[])
        seed_everything(seed)
        overall_frame_count = 0
        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)

        environments = self.__make_environments(environment=environment, seed=seed)
        observation_shape = environments.train.observation_space.shape
        self.__initialize_network(action_dimension=environments.train.action_space.n)
        self.__initialize_optimizer()

        train_obs, _ = environments.train.reset()
        test_obs, _ = environments.test.reset()
        observations = torch.from_numpy(
            numpy.concatenate([train_obs, test_obs], axis=0)
        ).to(self.device, non_blocking=True)
        epsilon_vector = torch.zeros(
            self.total_environments, dtype=torch.float32, device=self.device
        )

        buffer = PQNBuffer(
            steps_per_update=self.steps_per_update,
            total_environments=self.train_environments,
            observation_shape=observation_shape,
            action_dimension=environments.train.action_space.n,
            observation_dtype=torch.from_numpy(train_obs).dtype,
            device=self.device,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        for _ in range(self.total_updates):
            self._network.eval()
            for step in range(self.steps_per_update):
                epsilon_vector[: self.train_environments].fill_(
                    self._epsilon.get(
                        frames=overall_frame_count, total_frames=self.effective_frames
                    )
                )

                current_observations = observations
                q_values = self.__get_q_values(
                    observations=current_observations.float()
                )
                actions = self.__get_actions(
                    q_values=q_values, epsilon_vector=epsilon_vector
                )
                actions_numpy = actions.cpu().numpy()

                next_train_obs, train_reward, train_term, train_trunc, train_info = (
                    environments.train.step(actions_numpy[: self.train_environments])
                )
                next_test_obs, test_reward, test_term, test_trunc, test_info = (
                    environments.test.step(actions_numpy[self.train_environments :])
                )

                next_observations = numpy.concatenate(
                    [next_train_obs, next_test_obs], axis=0
                )
                train_terminations = numpy.logical_or(train_term, train_trunc)
                test_terminations = numpy.logical_or(test_term, test_trunc)
                terminations = numpy.concatenate(
                    [train_terminations, test_terminations], axis=0
                )

                info = self.__get_info(train_info, test_info)
                if "reward" in info:
                    episode_returns += info["reward"]

                if numpy.any(terminations):
                    for idx, score in zip(
                        numpy.where(terminations)[0], episode_returns[terminations]
                    ):
                        (
                            results.train
                            if idx < self.train_environments
                            else results.test
                        ).append(score)
                    episode_returns[terminations] = 0

                observations = torch.from_numpy(next_observations).to(
                    device=self.device, non_blocking=True
                )

                buffer.observations[step] = current_observations[
                    : self.train_environments
                ]
                buffer.actions[step] = actions[: self.train_environments]
                buffer.rewards[step] = torch.as_tensor(
                    train_reward, dtype=torch.float32, device=self.device
                )
                buffer.terminations[step] = torch.as_tensor(
                    train_terminations, dtype=torch.float32, device=self.device
                )
                buffer.q[step] = q_values[: self.train_environments]

                overall_frame_count += self.train_environments

            targets = self.__get_targets(
                buffer=buffer, observations=observations[: self.train_environments]
            )

            flat_obs = buffer.observations.contiguous().view((-1,) + observation_shape)
            flat_act = buffer.actions.contiguous().view(-1)
            flat_tgt = targets.contiguous().view(-1)

            self._network.train()
            batch_size = int(self.train_environments * self.steps_per_update)
            mini_size = batch_size // self.minibatches

            last_loss = None
            for _ in range(self.epochs):
                indices = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mini_size):
                    mini_idx = indices[start : start + mini_size]

                    self._optimizer.zero_grad(set_to_none=True)
                    loss = self.__get_loss(
                        observations=flat_obs[mini_idx],
                        actions=flat_act[mini_idx],
                        targets=flat_tgt[mini_idx],
                    )

                    scaler.scale(loss).backward()
                    scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), 10.0)
                    scaler.step(self._optimizer)
                    scaler.update()
                    last_loss = loss.detach()

            if last_loss is not None:
                results.loss.append(float(last_loss.cpu()))

        environments.train.close()
        environments.test.close()
        duration_seconds = time.perf_counter() - started_at
        self._results = results
        self._duration_seconds = duration_seconds
        return results, self._network

    @autocast()
    @torch.inference_mode()
    def __get_targets(
        self, observations: torch.Tensor, buffer: PQNBuffer
    ) -> torch.Tensor:
        next_q = self._network(observations.float()).max(dim=-1).values
        max_q_seq = buffer.q.max(dim=-1).values
        q_seq_for_lambda = torch.cat([max_q_seq, next_q.unsqueeze(0)])
        return lambda_returns(
            rewards=buffer.rewards,
            terminations=buffer.terminations,
            next_q=q_seq_for_lambda[1:],
            gamma=self.gamma,
            return_lambda=self.return_lambda,
        )

    @autocast()
    def __get_loss(
        self, observations: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        q_values_batch = self._network(observations.float())
        q_taken = q_values_batch.gather(1, actions.unsqueeze(1).long()).squeeze(-1)
        return 0.5 * mse_loss(q_taken, targets)

    def __get_info(self, train_info, test_info):
        info = {}
        for k, v_train in train_info.items():
            if (
                k in test_info
                and isinstance(v_train, numpy.ndarray)
                and isinstance(test_info[k], numpy.ndarray)
            ):
                v_test = test_info[k]
                info[k] = (
                    numpy.stack([v_train, v_test])
                    if v_train.ndim == 0
                    else numpy.concatenate([v_train, v_test], axis=0)
                )
        return info

    @autocast()
    @torch.inference_mode()
    def __get_actions(
        self, q_values: torch.Tensor, epsilon_vector: torch.Tensor
    ) -> torch.Tensor:
        if self.epsilon_greedy:
            return epsilon_greedy_vectorized(q_values, epsilon_vector)
        return q_values.argmax(dim=-1)

    @autocast()
    @torch.inference_mode()
    def __get_q_values(self, observations: torch.Tensor) -> torch.Tensor:
        return self._network(observations)

    def __create_directory(self, directory_path: str) -> str:
        path = directory_path.replace(".", "/").strip("/").strip()
        Path(path).mkdir(exist_ok=True, parents=True)
        return path

    def log(self, directory: str = "results") -> None:
        path = self.__create_directory(directory)
        duration_seconds = getattr(self, "_duration_seconds", None)

        results = results or getattr(self, "_results", None)
        if results is None:
            raise ValueError("No results to log. Run train() first or pass results=...")

        payload = {
            "duration_seconds": duration_seconds,
            "duration_hours": duration_seconds / 3600,
            "test_rewards": to_float_list(results.test),
            "train_rewards": to_float_list(results.train),
            "loss": to_float_list(results.loss),
        }

        funnel(f"{path}/result.json", payload)

    def save(self, *, directory: str = "models"):
        self.__create_directory(directory)
