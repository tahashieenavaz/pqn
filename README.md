# PQN

<div align="center">
    <img src="https://raw.githubusercontent.com/tahashieenavaz/pqn/main/images/pqn.png" width="95%" align="center" />
</div>

An custom implementation of PQN deep reinforcement learning algorithm.

## Installation

```bash
pip install pqn
```

## Usage

```py
from pqn import PQN

agent = PQN()
agent.train(environment="Pong-v5", seed=100)
agent.log(directory="results")
agent.save(directory="models")
```

## Abstract

Q-learning played a foundational role in the field reinforcement learning (RL).  However, TD algorithms with off-policy data, such as Q-learning, or nonlinear function approximation like deep neural networks require several additional tricks to stabilise training, primarily a large replay buffer and target networks.

Unfortunately, the delayed updating of frozen network parameters in the target network harms the sample efficiency and, similarly, the large replay buffer introduces memory and implementation overheads. In this paper, we investigate whether it is possible to accelerate and simplify off-policy TD training while maintaining its stability. Our key theoretical result demonstrates for the first time that regularisation techniques such as LayerNorm can yield provably convergent TD algorithms without the need for a target network or replay buffer, even with off-policy data.

Empirically, we find that online, parallelised sampling enabled by vectorised environments stabilises training without the need for a large replay buffer. Motivated by these findings, we propose PQN, our simplified deep online Q-Learning algorithm. Surprisingly, this simple algorithm is competitive with more complex methods like: Rainbow in Atari, PPO-RNN in Craftax, QMix in Smax, and can be up to 50x faster than traditional DQN without sacrificing sample efficiency. In an era where PPO has become the go-to RL algorithm, PQN reestablishes off-policy Q-learning as a viable alternative.


# Citation

```bibtex
@misc{2407.04811,
    Title = {Simplifying Deep Temporal Difference Learning},
    Author = {Matteo Gallici and Mattie Fellows and Benjamin Ellis and Bartomeu Pou and Ivan Masmitja and Jakob Nicolaus Foerster and Mario Martin},
    Year = {2024},
    Eprint = {arXiv:2407.04811},
}
```