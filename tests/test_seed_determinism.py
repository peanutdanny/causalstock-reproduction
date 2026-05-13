import numpy as np
import torch

from src.utils import load_yaml_config, set_global_seed


def test_torch_determinism():
    set_global_seed(0)
    a = torch.randn(8, 8)
    set_global_seed(0)
    b = torch.randn(8, 8)
    assert torch.equal(a, b)


def test_numpy_determinism():
    set_global_seed(42)
    a = np.random.rand(16)
    set_global_seed(42)
    b = np.random.rand(16)
    assert np.array_equal(a, b)


def test_config_loads_acl18():
    cfg = load_yaml_config("experiments/configs/acl18.yaml")
    assert cfg.model.d_p == 4
    assert cfg.model.d_m == 64
    assert cfg.train.lr == 1e-5
    assert cfg.data.lag_L == 5
    assert cfg.loss.bce_weight == 0.01
