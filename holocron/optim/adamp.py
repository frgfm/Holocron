# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from collections.abc import Callable, Iterable
from functools import lru_cache
from importlib.resources import files
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam

__all__ = ["AdamP", "adamp"]


class AdamP(Adam):
    r"""Implements the AdamP optimizer from ["AdamP: Slowing Down the Slowdown for Momentum Optimizers on
    Scale-invariant Weights"](https://arxiv.org/pdf/2006.08217.pdf).

    The estimation of momentums is described as follows, $\forall t \geq 1$:

    $$
    m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
    v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
    $$

    where $g_t$ is the gradient of $\theta_t$,
    $\beta_1, \beta_2 \in [0, 1]^2$ are the exponential average smoothing coefficients,
    $m_0 = g_0,\ v_0 = 0$.

    Then we correct their biases using:

    $$
    \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
    \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t}
    $$

    And finally the update step is performed using the following rule:

    $$
    p_t \leftarrow \frac{\hat{m_t}}{\sqrt{\hat{n_t} + \epsilon}} \\
    q_t \leftarrow \begin{cases}
      \prod_{\theta_t}(p_t) & if\ cos(\theta_t, g_t) < \delta / \sqrt{dim(\theta)}\\
      p_t & \text{otherwise}\\
    \end{cases} \\
    \theta_t \leftarrow \theta_{t-1} - \alpha q_t
    $$

    where $\theta_t$ is the parameter value at step $t$ ($\theta_0$ being the initialization value),
    $\prod_{\theta_t}(p_t)$ is the projection of $p_t$ onto the tangent space of $\theta_t$,
    $cos(\theta_t, g_t)$ is the cosine similarity between $\theta_t$ and $g_t$,
    $\alpha$ is the learning rate, $\delta > 0$, $\epsilon > 0$.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        betas: coefficients used for running averages
        eps: term added to the denominator to improve numerical stability
        weight_decay: weight decay (L2 penalty)
        amsgrad: whether to use the AMSGrad variant
        delta: delta threshold for projection
        mps_fused: opt into Metal fusion for contiguous FP32 MPS tensors without weight decay or AMSGrad;
            other configurations use the ordinary implementation
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        delta: float = 0.1,
        mps_fused: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.delta = delta
        self.mps_fused = mps_fused

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            loss value

        Raises:
            RuntimeError: if the optimizer does not support sparse gradients
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(f"{self.__class__.__name__} does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    if isinstance(state["step"], Tensor):
                        # Native Adam restores legacy integer steps as CPU tensors.
                        state["step"] = int(state["step"].item())
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            beta1, beta2 = group["betas"]
            adamp(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group["amsgrad"],
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
                self.delta,
                mps_fused=self.mps_fused,
            )

        return loss


def adamp(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    delta: float,
    *,
    mps_fused: bool = False,
) -> None:
    r"""Functional API that performs AdamP algorithm computation.
    See [`AdamP`][holocron.optim.AdamP] for details.
    """
    if mps_fused and _mps_adamp(
        params, grads, exp_avgs, exp_avg_sqs, state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps, delta
    ):
        return
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        # Extra step
        pt = exp_avg / bias_correction1 / denom
        project = F.cosine_similarity(param.data.reshape(1, -1), grad.reshape(1, -1)).max() < delta / math.sqrt(
            param.data.numel()
        )
        normalized_param = param.data / param.data.norm().add_(eps)
        # Tensor selection avoids synchronizing the accelerator once per parameter.
        pt = torch.where(project, pt - (normalized_param * pt).sum() * normalized_param, pt)

        param.add_(pt, alpha=-lr)


@lru_cache(maxsize=1)
def _mps_adamp_shader() -> Any | None:
    compiler = getattr(torch.mps, "compile_shader", None)
    if compiler is None:
        return None
    try:
        shader = compiler(files("holocron.optim").joinpath("adamp.metal").read_text(encoding="utf-8"))
        if any(
            getattr(kernel, "max_threads_per_threadgroup", 0) < 256 for kernel in (shader.moments, shader.statistics)
        ):
            return None
    except (RuntimeError, SyntaxError):
        # Compilation has not touched optimizer state; cache the ordinary fallback.
        return None
    return shader


def _mps_adamp(
    params: list[Tensor],
    grads: list[Tensor],
    avgs: list[Tensor],
    squares: list[Tensor],
    steps: list[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    delta: float,
) -> bool:
    if torch.is_grad_enabled() or amsgrad or not params:
        return False
    if any(isinstance(value, Tensor) for value in (beta1, beta2, lr, weight_decay, eps, delta)):
        return False
    if weight_decay != 0 or any(
        tensor.device.type != "mps"
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or tensor.is_neg()
        or not 0 < tensor.numel() < 2**32
        for tensors in zip(params, grads, avgs, squares, strict=True)
        for tensor in tensors
    ):
        return False
    if any(
        tensor.shape != tensors[0].shape
        for tensors in zip(params, grads, avgs, squares, strict=True)
        for tensor in tensors[1:]
    ) or any(
        len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != 4
        for tensors in zip(params, grads, avgs, squares, strict=True)
    ):
        return False
    shader = _mps_adamp_shader()
    if shader is None:
        return False
    # Metal bindings distinguish integer and float constants, including vector elements.
    beta1, beta2, lr, eps, delta = (float(value) for value in (beta1, beta2, lr, eps, delta))
    for param, grad, avg, square, step in zip(params, grads, avgs, squares, steps, strict=True):
        n = param.numel()
        groups = (n + 255) // 256
        update = torch.empty_like(param)
        partials = torch.empty((groups, 4), device=param.device, dtype=param.dtype)
        factor = torch.empty(2, device=param.device, dtype=param.dtype)
        shader.moments(
            update,
            param,
            grad,
            avg,
            square,
            partials,
            [beta1, beta2, 1 - beta1, 1 - beta2],
            [1 - beta1**step, math.sqrt(1 - beta2**step)],
            eps,
            n,
            threads=groups * 256,
            group_size=256,
        )
        torch.autograd.graph.increment_version((avg, square))
        shader.statistics(factor, partials, groups, n, eps, delta, threads=256, group_size=256)
        shader.apply_update(param, update, factor, lr)
        torch.autograd.graph.increment_version(param)
    return True
