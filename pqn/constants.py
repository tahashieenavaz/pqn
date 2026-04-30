import torch
from typing import Literal
from typing import Type

OptimizerStringType = Literal["radam", "nadam", "adamw", "nadam"]
PQNOptimizerType = OptimizerStringType | Type[torch.optim.Optimizer]
