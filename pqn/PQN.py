import torch
import numpy
import gymnasium
from dataclasses import dataclass
from baloot import acceleration_device
from .PQNNetwork import PQNNetwork


@dataclass
class Agent:
    num_environments: int = 128
    activation_function = torch.nn.GELU
    learning_rate: float = 0.00025
    timesteps: int = 10_000_000
    num_steps: int = 32
    num_epochs: int = 2
    num_minibatches: int = 32
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.001
    gamma: float = 0.99
    lamb: float = 0.65
    t: int = 0

    def __init__(self, environment_name: str):
        self.device = acceleration_device()
        self.environments = self.make_environments(
            environment_name, self.num_environments
        )
        self.network = PQNNetwork(
            action_dimension=self.environments.action_space.n,
            activation_function=self.activation_function,
        ).to(self.device)

        self.optimizer = torch.optim.RAdam(
            self.network.parameters(), lr=self.learning_rate, eps=1e-5
        )

        self.states = numpy.zeros(
            shape=(self.num_steps, self.num_environments, 84, 84), dtype=numpy.uint8
        )
        self.next_states = numpy.zeros(
            shape=(self.num_steps, self.num_environments, 84, 84), dtype=numpy.uint8
        )
        self.actions = numpy.zeros(
            shape=(self.num_steps, self.num_environments), dtype=numpy.int64
        )
        self.rewards = numpy.zeros(
            shape=(self.num_steps, self.num_environments), dtype=numpy.float32
        )
        self.terminations = numpy.zeros(
            shape=(self.num_steps, self.num_environments), dtype=numpy.float32
        )

    @property
    def lr(self):
        return self.learning_rate

    def epsilon(self):
        pass

    def make_environments(self, name: str, count: int):
        envs = gymnasium.vector.SyncVectorEnv(
            [lambda: gymnasium.make(name, frameskip=1) for _ in range(count)]
        )
        envs = gymnasium.wrappers.AtariPreprocessing(
            grayscale_obs=True, scale_obs=False, frame_skip=4, screen_size=84
        )
        envs = gymnasium.wrappers.FrameStackObservation(envs, 4)
        return envs

    def tick(self):
        self.t += self.num_environments

    def loop(self):
        _episode = 0
        while self.t < self.timesteps:
            episode_loss = 0.0
            episode_reward = 0.0

            for step in range(self.num_steps):
                self.tick()
                torch_state = torch.from_numpy(state)
                actions = self.actions(torch_state)

    def train(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()

    def loss(self):
        pass

    def actions(self, states):
        _device = self.device
        q = self.network(states)
        greedy_actions = q.argmax(dim=-1)
        random_actions = torch.randint(
            0,
            self.environments.action_space.n,
            size=greedy_actions.shape,
            device=_device,
        )
        is_exploring = torch.rand(greedy_actions.shape, device=_device) < self.epsilon
        return torch.where(is_exploring, random_actions, greedy_actions)
