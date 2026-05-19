import torch
import numpy
import envpool
import os
import time
from contextlib import nullcontext
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

        if self.train_environments <= 0:
            raise ValueError("train_environments must be greater than zero")
        if self.test_environments < 0:
            raise ValueError("test_environments must be greater than or equal to zero")

        torch.set_float32_matmul_precision("high")
        self.device = acceleration_device()
        self._epsilon = LinearEpsilon()

        self.total_environments = self.train_environments + self.test_environments
        self.total_cpu = os.cpu_count() or 1
        self.has_test_environments = self.test_environments > 0
        if self.has_test_environments:
            requested_train_cpu = int(self.train_cpu_distribution * self.total_cpu)
            if self.total_cpu == 1:
                self.train_cpu_count = 1
                self.test_cpu_count = 1
            else:
                self.train_cpu_count = min(
                    max(1, requested_train_cpu), self.total_cpu - 1
                )
                self.test_cpu_count = max(1, self.total_cpu - self.train_cpu_count)
        else:
            self.train_cpu_count = self.total_cpu
            self.test_cpu_count = 0
        self.effective_frames = int(self.frames / self.frame_skip)
        self.total_updates = int(
            self.effective_frames / self.train_environments / self.steps_per_update
        )
        if (self.train_environments * self.steps_per_update) % self.minibatches != 0:
            raise ValueError(
                "minibatches must divide train_environments * steps_per_update"
            )

    def __autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    @torch.inference_mode()
    def __initialize_network(
        self,
        *,
        action_dimension: int,
        observation_shape: tuple,
        observation_dtype: torch.dtype,
    ):
        self._network = network_map[self.network](action_dimension=action_dimension).to(
            self.device
        )
        if self.device.type == "cuda":
            self._network = torch.compile(self._network, mode="reduce-overhead")

        warmup_batches = {
            self.train_environments,
            int(self.train_environments * self.steps_per_update / self.minibatches),
        }
        if self.has_test_environments:
            warmup_batches.add(self.total_environments)

        with self.__autocast_context():
            for batch_size in sorted(batch for batch in warmup_batches if batch > 0):
                self._network(
                    torch.zeros(
                        (batch_size,) + observation_shape,
                        dtype=observation_dtype,
                        device=self.device,
                    )
                )

    def __initialize_optimizer(self):
        self._optimizer = optimizer_map[self.optimizer](
            self._network.parameters(),
            lr=self.lr,
            eps=self.optimizer_epsilon,
            betas=(self.optimizer_first_beta, self.optimizer_second_beta),
            weight_decay=self.optimizer_weight_decay,
        )

    def __environment_factory(
        self,
        environment: str,
        environments: int,
        cpu_count: int,
        life: bool,
        seed: int,
        thread_affinity_offset: int,
    ):
        return envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=environments,
            seed=seed,
            num_threads=cpu_count,
            thread_affinity_offset=thread_affinity_offset,
            noop_max=self.noop,
            frame_skip=self.frame_skip,
            repeat_action_probability=0.0,
            reward_clip=True,
            episodic_life=life,
        )

    def __make_environments(self, environment: str, seed: int):
        train = self.__environment_factory(
            environment=environment,
            environments=self.train_environments,
            cpu_count=self.train_cpu_count,
            life=True,
            seed=seed,
            thread_affinity_offset=0,
        )
        test = None
        if self.has_test_environments:
            test = self.__environment_factory(
                environment=environment,
                environments=self.test_environments,
                cpu_count=self.test_cpu_count,
                life=False,
                seed=seed + 1000,
                thread_affinity_offset=(
                    self.train_cpu_count if self.total_cpu > 1 else 0
                ),
            )
        return SimpleNamespace(
            train=train,
            test=test,
        )

    def train(self, *, environment: str, seed: int):
        started_at = time.perf_counter()
        results = SimpleNamespace(loss=[], test=[], train=[])
        seed_everything(seed)
        overall_frame_count = 0
        train_episode_returns = numpy.zeros(
            self.train_environments, dtype=numpy.float32
        )
        test_episode_returns = numpy.zeros(self.test_environments, dtype=numpy.float32)

        environments = self.__make_environments(environment=environment, seed=seed)
        observation_shape = environments.train.observation_space.shape

        train_obs, _ = environments.train.reset()
        test_obs = None
        if environments.test is not None:
            test_obs, _ = environments.test.reset()

        observation_dtype = torch.from_numpy(train_obs).dtype
        self.__initialize_network(
            action_dimension=environments.train.action_space.n,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
        )
        self.__initialize_optimizer()

        train_observations = torch.from_numpy(train_obs).to(
            self.device, non_blocking=True
        )
        test_observations = (
            torch.from_numpy(test_obs).to(self.device, non_blocking=True)
            if test_obs is not None
            else None
        )
        train_epsilon = torch.empty(
            self.train_environments, dtype=torch.float32, device=self.device
        )

        buffer = PQNBuffer(
            steps_per_update=self.steps_per_update,
            total_environments=self.train_environments,
            observation_shape=observation_shape,
            action_dimension=environments.train.action_space.n,
            observation_dtype=observation_dtype,
            device=self.device,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        batch_size = int(self.train_environments * self.steps_per_update)
        mini_size = batch_size // self.minibatches

        for _ in range(self.total_updates):
            self._network.eval()
            for step in range(self.steps_per_update):
                train_epsilon.fill_(
                    self._epsilon.get(
                        frames=overall_frame_count, total_frames=self.effective_frames
                    )
                )

                current_train_observations = train_observations
                if test_observations is not None:
                    q_values = self.__get_q_values(
                        observations=torch.cat(
                            (current_train_observations, test_observations), dim=0
                        )
                    )
                    train_q_values = q_values[: self.train_environments]
                    test_q_values = q_values[self.train_environments :]
                else:
                    train_q_values = self.__get_q_values(
                        observations=current_train_observations
                    )
                    test_q_values = None

                train_actions = self.__get_actions(
                    q_values=train_q_values, epsilon_vector=train_epsilon
                )

                next_train_obs, train_reward, train_term, train_trunc, train_info = (
                    environments.train.step(train_actions.cpu().numpy())
                )

                if environments.test is not None:
                    test_actions = test_q_values.argmax(dim=-1)
                    next_test_obs, test_reward, test_term, test_trunc, test_info = (
                        environments.test.step(test_actions.cpu().numpy())
                    )
                    test_terminations = numpy.logical_or(test_term, test_trunc)
                    test_episode_returns += test_info.get("reward", test_reward)
                    if numpy.any(test_terminations):
                        results.test.extend(
                            test_episode_returns[test_terminations].tolist()
                        )
                        test_episode_returns[test_terminations] = 0
                    test_observations = torch.from_numpy(next_test_obs).to(
                        device=self.device, non_blocking=True
                    )

                train_terminations = numpy.logical_or(train_term, train_trunc)
                train_episode_returns += train_info.get("reward", train_reward)
                if numpy.any(train_terminations):
                    results.train.extend(
                        train_episode_returns[train_terminations].tolist()
                    )
                    train_episode_returns[train_terminations] = 0

                train_observations = torch.from_numpy(next_train_obs).to(
                    device=self.device, non_blocking=True
                )

                buffer.observations[step] = current_train_observations
                buffer.actions[step] = train_actions
                buffer.rewards[step] = torch.as_tensor(
                    train_reward, dtype=torch.float32, device=self.device
                )
                buffer.terminations[step] = torch.as_tensor(
                    train_terminations, dtype=torch.float32, device=self.device
                )
                buffer.q[step] = train_q_values

                overall_frame_count += self.train_environments

            targets = self.__get_targets(
                buffer=buffer, observations=train_observations
            )

            flat_obs = buffer.observations.contiguous().view((-1,) + observation_shape)
            flat_act = buffer.actions.contiguous().view(-1)
            flat_tgt = targets.contiguous().view(-1)

            self._network.train()

            last_loss = None
            for _ in range(self.epochs):
                indices = torch.randperm(batch_size, device=self.device)
                shuffled_obs = flat_obs.index_select(0, indices)
                shuffled_act = flat_act.index_select(0, indices)
                shuffled_tgt = flat_tgt.index_select(0, indices)
                for start in range(0, batch_size, mini_size):
                    stop = start + mini_size

                    self._optimizer.zero_grad(set_to_none=True)
                    loss = self.__get_loss(
                        observations=shuffled_obs[start:stop],
                        actions=shuffled_act[start:stop],
                        targets=shuffled_tgt[start:stop],
                    )

                    scaler.scale(loss).backward()
                    scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), 10.0)
                    scaler.step(self._optimizer)
                    scaler.update()
                    last_loss = loss.detach()

            if last_loss is not None:
                results.loss.append(last_loss)

        environments.train.close()
        if environments.test is not None:
            environments.test.close()
        if results.loss:
            results.loss = torch.stack(results.loss).detach().cpu().tolist()
        duration_seconds = time.perf_counter() - started_at
        self._results = results
        self._duration_seconds = duration_seconds
        return results, self._network

    @autocast()
    @torch.inference_mode()
    def __get_targets(
        self, observations: torch.Tensor, buffer: PQNBuffer
    ) -> torch.Tensor:
        next_q = self._network(observations).max(dim=-1).values
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
        q_values_batch = self._network(observations)
        q_taken = q_values_batch.gather(1, actions.unsqueeze(1).long()).squeeze(-1)
        return 0.5 * mse_loss(q_taken, targets)

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

        results = getattr(self, "_results", None)
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
