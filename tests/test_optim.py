import importlib
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

from holocron import optim


def _test_optimizer(name: str, **kwargs: Any) -> None:
    lr = 1e-4
    input_shape = (3, 224, 224)
    num_batches = 4
    # Get model and optimizer
    model = mobilenet_v3_small(num_classes=10)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier[3].parameters():
        p.requires_grad_(True)
    optimizer = optim.__dict__[name](model.classifier[3].parameters(), lr=lr, **kwargs)

    # Save param value
    p_ = model.classifier[3].weight
    p_val = p_.data.clone()

    # Random inputs
    input_t = torch.rand((num_batches, *input_shape), dtype=torch.float32)
    target = torch.zeros(num_batches, dtype=torch.long)

    # Update
    optimizer.zero_grad()
    output = model(input_t)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    # Test
    assert p_.grad is not None
    assert not torch.equal(p_.data, p_val)


def test_lars():
    _test_optimizer("LARS", momentum=0.9, weight_decay=2e-5)


def test_lamb():
    _test_optimizer("LAMB", weight_decay=2e-5)


def test_ralars():
    _test_optimizer("RaLars", weight_decay=2e-5)


def test_tadam():
    _test_optimizer("TAdam")


def test_adabelief():
    _test_optimizer("AdaBelief")


def test_adamp(monkeypatch):
    _test_optimizer("AdamP")
    module = importlib.import_module("holocron.optim.adamp")
    monkeypatch.setattr(module, "_mps_adamp_shader", lambda: pytest.fail("CPU must not compile Metal"))
    _test_optimizer("AdamP", mps_fused=True)


def test_adamp_shader_unavailable_is_cached(monkeypatch):
    module = importlib.import_module("holocron.optim.adamp")
    module._mps_adamp_shader.cache_clear()
    try:
        for error in (RuntimeError, SyntaxError, OSError):
            compiler = Mock(side_effect=error("Metal compiler unavailable"))
            monkeypatch.setattr(torch.mps, "compile_shader", compiler, raising=False)
            assert module._mps_adamp_shader() is None
            assert module._mps_adamp_shader() is None
            compiler.assert_called_once()
            module._mps_adamp_shader.cache_clear()
    finally:
        module._mps_adamp_shader.cache_clear()


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Metal shaders unavailable",
)
def test_adamp_mps_fused_parity_and_versions(monkeypatch):
    module = importlib.import_module("holocron.optim.adamp")
    native_adamp = module._mps_adamp

    def require_fusion(*args):
        assert native_adamp(*args)
        return True

    monkeypatch.setattr(module, "_mps_adamp", require_fusion)
    cases = [
        (range(257), [1.0, -1.0] * 128 + [1.0]),
        ([0.2, 0.1], [0.5093396902084351, -0.8605654835700989]),
        ([3e38, 3e38], [1e-5, 1e-5]),
    ]
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for values, gradients in cases:
            params = [torch.nn.Parameter(torch.tensor(values, device="mps", dtype=torch.float32)) for _ in range(2)]
            reference, candidate = [
                optim.AdamP([param], lr=1, betas=(0.0, 0.0), mps_fused=fused)
                for param, fused in zip(params, (False, True), strict=True)
            ]
            for param in params:
                param.grad = torch.tensor(gradients, device="mps", dtype=torch.float32)
            reference.step()
            candidate.step()
            torch.testing.assert_close(*params, rtol=1e-5, atol=1e-6)
            assert params[1]._version > 0
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(reference.state[params[0]][key], candidate.state[params[1]][key])
                assert candidate.state[params[1]][key]._version > 0
            candidate.load_state_dict(deepcopy(candidate.state_dict()))
            reference.step()
            candidate.step()
            torch.testing.assert_close(*params, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_default_dtype(default_dtype)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Metal shaders unavailable",
)
def test_adamp_mps_fallback_and_amp_skip(monkeypatch):
    module = importlib.import_module("holocron.optim.adamp")
    parameter = torch.nn.Parameter(torch.ones(2, device="mps"))
    optimizer = optim.AdamP([parameter], mps_fused=True)
    scaler = torch.amp.GradScaler("mps", init_scale=128.0)
    scaler.scale(parameter.square().sum() * float("inf")).backward()
    scaler.step(optimizer)
    scaler.update()
    assert not optimizer.state
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    optimizer.zero_grad()
    scaler.scale(parameter.square().sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert optimizer.state[parameter]["step"] == 1
    assert torch.isfinite(parameter).all()
    optimizer.state[parameter]["exp_avg"] = parameter.grad
    with monkeypatch.context() as patch:
        patch.setattr(module, "_mps_adamp_shader", lambda: pytest.fail("Aliased state must fall back"))
        optimizer.step()
    optimizer.state[parameter]["exp_avg"] = parameter.grad.clone()
    shader = module._mps_adamp_shader()
    guarded = torch.ones(8, device="mps")
    shader.apply_update(
        guarded[:7], torch.ones(7, device="mps"), torch.tensor([0.0, 0.0, 1.0], device="mps"), 1.0, 7, threads=256
    )
    torch.testing.assert_close(guarded, torch.tensor([0.0] * 7 + [1.0], device="mps"))
    shader = Mock()
    shader.moments.side_effect = RuntimeError("dispatch failed")
    monkeypatch.setattr(module, "_mps_adamp_shader", lambda: shader)
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="dispatch failed"):
        optimizer.step()
    shader.moments.assert_called_once()
    shader.statistics.assert_not_called()
    torch.testing.assert_close(parameter, before)


def test_adan():
    _test_optimizer("Adan")


def test_adan_betas_and_state():
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    betas = (0.8, 0.9, 0.95)
    optimizer = optim.Adan([
        {"params": [params[0]]},
        {"params": [params[1]], "betas": betas, "amsgrad": True},
    ])
    assert optimizer.param_groups[0]["betas"] == (0.98, 0.92, 0.99)
    assert optimizer.param_groups[1]["betas"] == betas
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    assert all(torch.all(param < 1) for param in params)
    assert "max_exp_avg_delta" in optimizer.state[params[1]]

    restored_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    restored = optim.Adan([{"params": [param]} for param in restored_params])
    state = deepcopy(optimizer.state_dict())
    del state["param_groups"][0]["amsgrad"]
    state["state"][0]["step"] = torch.tensor(state["state"][0]["step"])
    restored.load_state_dict(state)
    assert restored.param_groups[0]["amsgrad"] is False
    assert restored.param_groups[1]["betas"] == betas
    assert restored.param_groups[1]["amsgrad"] is True
    for param in restored_params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    restored.step()
    for actual, expected in zip(restored_params, params, strict=True):
        torch.testing.assert_close(actual, expected)

    for kwargs in (
        {"lr": -1.0},
        {"eps": -1.0},
        {"weight_decay": -1.0},
        {"betas": (0.9, 0.99)},
        {"betas": (0.8, 0.9, 1.0)},
        {"betas": (0.8, -0.1, 0.99)},
    ):
        with pytest.raises(ValueError):
            optim.Adan(params, **kwargs)


def test_ademamix():
    _test_optimizer("AdEMAMix")
