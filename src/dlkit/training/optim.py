from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import torch.optim as optim


class OptimizerType(StrEnum):  # TODO: add new optimizers as needed
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


OPTIMIZERS: dict[OptimizerType, Callable[..., optim.Optimizer]] = {
    OptimizerType.ADAM: optim.Adam,
    OptimizerType.ADAMW: optim.AdamW,
    OptimizerType.SGD: lambda params, lr, weight_decay, *, momentum=0.9: optim.SGD(
        params, lr=lr, weight_decay=weight_decay, momentum=momentum
    ),
}
