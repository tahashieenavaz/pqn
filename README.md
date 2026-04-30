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
