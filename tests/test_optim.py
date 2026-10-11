import importlib
import math
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


def test_adamp():
    _test_optimizer("AdamP")


def test_adamp_fused_cpu_fallback_and_resume(monkeypatch):
    module = importlib.import_module("holocron.optim.adamp")
    monkeypatch.setattr(module, "_mps_adamp_shader", lambda: pytest.fail("CPU must not compile Metal"))
    for kwargs in ({}, {"weight_decay": 0.1}, {"amsgrad": True}, {"betas": (0.95, 0.99)}):
        params = [torch.nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(2, 3).T)]
        copies = [torch.nn.Parameter(param.detach().clone()) for param in params]
        reference = optim.AdamP(params, **kwargs)
        candidate = optim.AdamP(copies, mps_fused=True, **kwargs)
        for param in (*params, *copies):
            param.grad = torch.ones_like(param)
        reference.step()
        candidate.step()
        candidate.load_state_dict(deepcopy(candidate.state_dict()))
        reference.step()
        candidate.step()
        torch.testing.assert_close(params[0], copies[0])
        assert candidate.state[copies[0]]["step"] == 2
        assert isinstance(candidate.state[copies[0]]["step"], int)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(reference.state[params[0]][key], candidate.state[copies[0]][key])


def test_adamp_shader_unavailable_is_cached(monkeypatch):
    module = importlib.import_module("holocron.optim.adamp")
    module._mps_adamp_shader.cache_clear()
    try:
        for error in (RuntimeError, SyntaxError):
            compiler = Mock(side_effect=error("Metal compiler unavailable"))
            monkeypatch.setattr(torch.mps, "compile_shader", compiler, raising=False)
            assert module._mps_adamp_shader() is None
            assert module._mps_adamp_shader() is None
            compiler.assert_called_once()
            module._mps_adamp_shader.cache_clear()
        monkeypatch.delattr(torch.mps, "compile_shader")
        assert module._mps_adamp_shader() is None
        module._mps_adamp_shader.cache_clear()
        shader = Mock()
        shader.moments.max_threads_per_threadgroup = 128
        shader.statistics.max_threads_per_threadgroup = 256
        monkeypatch.setattr(torch.mps, "compile_shader", Mock(return_value=shader), raising=False)
        assert module._mps_adamp_shader() is None
        module._mps_adamp_shader.cache_clear()
        with monkeypatch.context() as patch:
            resource = Mock()
            resource.joinpath.return_value.read_text.side_effect = OSError("Shader file unavailable")
            patch.setattr(module, "files", Mock(return_value=resource))
            compiler = Mock()
            patch.setattr(torch.mps, "compile_shader", compiler)
            assert module._mps_adamp_shader() is None
            assert module._mps_adamp_shader() is None
            compiler.assert_not_called()
            resource.joinpath.return_value.read_text.assert_called_once()
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
    torch.manual_seed(9)
    for betas, delta in (((0.9, 0.999), 0.0), ((0.95, 0.99), 10.0)):
        params = [
            torch.nn.Parameter(torch.randn(shape, device="mps"))
            for shape in ((), (7,), (2, 3), (17, 3, 3, 3), (66001,))
        ]
        params.append(torch.nn.Parameter(torch.zeros(7, device="mps")))
        copies = [torch.nn.Parameter(param.detach().clone()) for param in params]
        reference = optim.AdamP(params, betas=betas, delta=delta, eps=1e-6)
        candidate = optim.AdamP(copies, betas=betas, delta=delta, eps=1e-6, mps_fused=True)
        for step in range(8):
            for first, second in zip(params, copies, strict=True):
                first.grad = (torch.randn_like(first), torch.zeros_like(first), torch.full_like(first, 1e-12))[step % 3]
                second.grad = first.grad.clone()
            versions = [
                (second._version, tuple(candidate.state[second][key]._version for key in ("exp_avg", "exp_avg_sq")))
                if candidate.state[second]
                else (second._version, (0, 0))
                for second in copies
            ]
            reference.step()
            candidate.step()
            for first, second, (param_version, state_versions) in zip(params, copies, versions, strict=True):
                torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
                assert second._version > param_version
                for key, version in zip(("exp_avg", "exp_avg_sq"), state_versions, strict=True):
                    state = candidate.state[second][key]
                    torch.testing.assert_close(reference.state[first][key], state, rtol=1e-5, atol=1e-6)
                    assert state._version > version
            if step == 3:
                candidate.load_state_dict(deepcopy(candidate.state_dict()))
    cosines = (0.1 / math.sqrt(2) + offset for offset in (-1e-4, 0.0, 1e-4))
    cases = [
        ([0.2, 0.1], [0.5093396902084351, -0.8605654835700989]),
        ([3e38, 3e38], [1e-5, 1e-5]),
    ] + [([1.0, 0.0], [cosine, math.sqrt(1 - cosine**2)]) for cosine in cosines]
    default_dtype = torch.get_default_dtype()
    try:
        for dtype in (torch.float16, torch.float64):
            torch.set_default_dtype(dtype)
            for values, gradients in cases:
                params = [torch.nn.Parameter(torch.tensor(values, device="mps", dtype=torch.float32)) for _ in range(2)]
                for param in params:
                    param.grad = torch.tensor(gradients, device="mps", dtype=torch.float32)
                for param, fused in zip(params, (False, True), strict=True):
                    optim.AdamP([param], lr=1, betas=(0.0, 0.0), mps_fused=fused).step()
                torch.testing.assert_close(*params, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_default_dtype(default_dtype)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Metal shaders unavailable",
)
def test_adamp_mps_fallback_and_amp_skip(monkeypatch):
    module = importlib.import_module("holocron.optim.adamp")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_mps_adamp_shader", lambda: pytest.fail("Unsupported groups must not compile Metal"))
        for kwargs, tensor in (
            ({"weight_decay": 0.1}, torch.ones(2, device="mps")),
            ({"amsgrad": True}, torch.ones(2, device="mps")),
            ({"lr": torch.tensor(1e-3)}, torch.ones(2, device="mps")),
            ({}, torch.ones(2, device="mps", dtype=torch.float16)),
            ({}, torch.ones(2, 3, device="mps").T),
            ({}, torch.ones(2)),
        ):
            groups = [
                [torch.nn.Parameter(torch.ones(2, device="mps")), torch.nn.Parameter(tensor.clone())] for _ in range(2)
            ]
            for params, fused in zip(groups, (False, True), strict=True):
                for param in params:
                    param.grad = torch.ones_like(param)
                optim.AdamP(params, mps_fused=fused, **kwargs).step()
            for first, second in zip(*groups, strict=True):
                torch.testing.assert_close(first, second)
        for fused in (False, True):
            parameter = torch.nn.Parameter(torch.ones(2, 3, device="mps"))
            optimizer = optim.AdamP([parameter], mps_fused=fused)
            state = optimizer.state_dict()
            state["state"][0] = {
                "step": 0,
                "exp_avg": torch.zeros(1, device="mps"),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
            optimizer.load_state_dict(state)
            parameter.grad = torch.ones_like(parameter)
            with pytest.raises(RuntimeError):
                optimizer.step()
        params = [torch.nn.Parameter(torch.ones(2, device="mps")) for _ in range(2)]
        for parameter, fused in zip(params, (False, True), strict=True):
            parameter.grad = torch.ones_like(parameter)
            optimizer = optim.AdamP([parameter], mps_fused=fused)
            optimizer.state[parameter] = {
                "step": 0,
                "exp_avg": parameter.grad,
                "exp_avg_sq": torch.zeros_like(parameter),
            }
            optimizer.step()
        torch.testing.assert_close(*params)
    parameter = torch.nn.Parameter(torch.ones(2, device="mps"))
    optimizer = optim.AdamP([parameter], mps_fused=True)
    assert not getattr(optimizer, "_step_supports_amp_scaling", False)
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
    with pytest.raises(RuntimeError, match="dispatch failed"):
        optimizer.step()
    shader.moments.assert_called_once()
    shader.statistics.assert_not_called()


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
