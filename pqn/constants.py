import torch
from typing import Literal
from typing import Type

OptimizerStringType = Literal["radam", "nadam", "adamw", "nadam"]
NetworkStringType = Literal["q", "duelling"]
PQNOptimizerType = OptimizerStringType | Type[torch.optim.Optimizer]
