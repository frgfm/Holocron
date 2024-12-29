# Copyright (C) 2022-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Adam

__all__ = ["Adan", "adan"]


class Adan(Adam):
    r"""Implements the Adan optimizer from `"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep
    Models" <https://arxiv.org/pdf/2208.06677.pdf>`_.

    The estimation of momentums is described as follows, :math:`\forall t \geq 1`:

    .. math::
        m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) (g_t - g_{t-1}) \\
        n_t \leftarrow \beta_3 n_{t-1} + (1 - \beta_3) [g_t + \beta_2 (g_t - g_{t - 1})]^2

    where :math:`g_t` is the gradient of :math:`\theta_t`,
    :math:`\beta_1, \beta_2, \beta_3 \in [0, 1]^3` are the exponential average smoothing coefficients,
    :math:`m_0 = g_0,\ v_0 = 0,\ n_0 = g_0^2`.

    Then we correct their biases using:

    .. math::
        \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
        \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t} \\
        \hat{n_t} \leftarrow \frac{n_t}{1 - \beta_3^t}

    And finally the update step is performed using the following rule:

    .. math::
        p_t \leftarrow \frac{\hat{m_t} + (1 - \beta_2) \hat{v_t}}{\sqrt{\hat{n_t} + \epsilon}} \\
        \theta_t \leftarrow \frac{\theta_{t-1} - \alpha p_t}{1 + \lambda \alpha}

    where :math:`\theta_t` is the parameter value at step :math:`t` (:math:`\theta_0` being the initialization value),
    :math:`\alpha` is the learning rate, :math:`\lambda \geq 0` is the weight decay, :math:`\epsilon > 0`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float, float], optional): coefficients used for running averages
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)  # type: ignore[arg-type]

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            prev_grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_deltas = []
            max_exp_avg_deltas = []
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
                        # Exponential moving average of gradient delta values
                        state["exp_avg_delta"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_delta"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["prev_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    prev_grads.append(state["prev_grad"])
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avg_deltas.append(state["exp_avg_delta"])
                    if group["amsgrad"]:
                        max_exp_avg_deltas.append(state["max_exp_avg_delta"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            beta1, beta2, beta3 = group["betas"]
            adan(
                params_with_grad,
                grads,
                prev_grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_deltas,
                max_exp_avg_deltas,
                state_steps,
                group["amsgrad"],
                beta1,
                beta2,
                beta3,
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )

        return loss


def adan(
    params: List[Tensor],
    grads: List[Tensor],
    prev_grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_deltas: List[Tensor],
    max_exp_avg_deltas: List[Tensor],
    state_steps: List[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> None:
    r"""Functional API that performs Adan algorithm computation.
    See :class:`~holocron.optim.Adan` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_delta = exp_avg_deltas[i]
        prev_grad = prev_grads[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        bias_correction3 = 1 - beta3**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        delta_grad = grad - prev_grad
        exp_avg_sq.mul_(beta2).add_(delta_grad, alpha=1 - beta2)
        _tmp = grad + beta2 * delta_grad
        exp_avg_delta.mul_(beta3).addcmul_(_tmp, _tmp, value=1 - beta3)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_deltas[i], exp_avg_delta, out=max_exp_avg_deltas[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_deltas[i].sqrt() / math.sqrt(bias_correction3)).add_(eps)
        else:
            denom = (exp_avg_delta.sqrt() / math.sqrt(bias_correction3)).add_(eps)

        # Extra step
        pt = (exp_avg / bias_correction1 + beta2 * exp_avg_sq / bias_correction2) / denom

        param.add_(pt, alpha=-lr)
        if weight_decay != 0:
            param /= 1 + weight_decay * lr
