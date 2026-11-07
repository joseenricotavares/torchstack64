import torch
from dlkit.training.optim import OPTIMIZERS, OptimizerType


def _params():
    return [torch.nn.Parameter(torch.zeros(2))]


def test_adam_and_adamw_receive_lr_and_weight_decay():
    cases = [(OptimizerType.ADAM, torch.optim.Adam), (OptimizerType.ADAMW, torch.optim.AdamW)]
    for kind, expected_cls in cases:
        optimizer = OPTIMIZERS[kind](_params(), lr=1e-3, weight_decay=1e-2)
        assert isinstance(optimizer, expected_cls)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 1e-2


def test_sgd_defaults_to_momentum_point_nine():
    optimizer = OPTIMIZERS[OptimizerType.SGD](_params(), lr=1e-2, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9


def test_sgd_accepts_explicit_momentum_override():
    optimizer = OPTIMIZERS[OptimizerType.SGD](_params(), lr=1e-2, weight_decay=0.0, momentum=0.5)
    assert optimizer.defaults["momentum"] == 0.5
