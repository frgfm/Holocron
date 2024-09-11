# Copyright (C) 2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer  # type: ignore[attr-defined]

__all__ = ["AdEMAMix", "ademamix"]


class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix optimizer from `"The AdEMAMix Optimizer: Better, Faster, Older" <https://arxiv.org/pdf/2409.03137>`_.

    The estimation of momentums is described as follows, :math:`\forall t \geq 1`:

    .. math::
        m_{1,t} \leftarrow \beta_1 m_{1, t-1} + (1 - \beta_1) g_t \\
        m_{2,t} \leftarrow \beta_3 m_{2, t-1} + (1 - \beta_3) g_t \\
        s_t \leftarrow \beta_2 s_{t-1} + (1 - \beta_2) (g_t - m_t)^2 + \epsilon

    where :math:`g_t` is the gradient of :math:`\theta_t`,
    :math:`\beta_1, \beta_2, \beta_3 \in [0, 1]^3` are the exponential average smoothing coefficients,
    :math:`m_{1,0} = 0,\ m_{2,0} = 0,\ s_0 = 0`, :math:`\epsilon > 0`.

    Then we correct their biases using:

    .. math::
        \hat{m_{1,t}} \leftarrow \frac{m_{1,t}}{1 - \beta_1^t} \\
        \hat{s_t} \leftarrow \frac{s_t}{1 - \beta_2^t}

    And finally the update step is performed using the following rule:

    .. math::
        \theta_t \leftarrow \theta_{t-1} - \eta \frac{\hat{m_{1,t}} + \alpha m_{2,t}}{\sqrt{\hat{s_t}} + \epsilon}

    where :math:`\theta_t` is the parameter value at step :math:`t` (:math:`\theta_0` being the initialization value),
    :math:`\eta` is the learning rate, :math:`\alpha > 0` :math:`\epsilon > 0`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float, float], optional): coefficients used for running averages (default: (0.9, 0.999, 0.9999))
        alpha (float, optional): the exponential decay rate of the second moment estimates (default: 5.0)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        for idx, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {idx}: {beta}")
        defaults = {"lr": lr, "betas": betas, "alpha": alpha, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avgs_slow = []
            exp_avg_sqs = []
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
                        state["exp_avg_slow"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avgs_slow.append(state["exp_avg_slow"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            beta1, beta2, beta3 = group["betas"]
            ademamix(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avgs_slow,
                exp_avg_sqs,
                state_steps,
                beta1,
                beta2,
                beta3,
                group["alpha"],
                group["lr"],
                group["weight_decay"],
                group["eps"],
            )
        return loss


def ademamix(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avgs_slow: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> None:
    r"""Functional API that performs AdaBelief algorithm computation.
    See :class:`~holocron.optim.AdaBelief` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        m1 = exp_avgs[i]
        m2 = exp_avgs_slow[i]
        nu = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        m1.mul_(beta1).add_(grad, alpha=1 - beta1)
        nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        m2.mul_(beta3).add_(grad, alpha=1 - beta3)

        denom = (nu.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        param.addcdiv_(m1 / bias_correction1 + alpha * m2, denom, value=-lr)
