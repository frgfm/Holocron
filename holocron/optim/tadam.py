# Copyright (C) 2019-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["TAdam", "tadam"]


class TAdam(Optimizer):
    r"""Implements the TAdam optimizer from `"TAdam: A Robust Stochastic Gradient Optimizer"
    <https://arxiv.org/pdf/2003.00179.pdf>`_.

    The estimation of momentums is described as follows, :math:`\forall t \geq 1`:

    .. math::
        w_t \leftarrow (\nu + d) \Big(\nu + \sum\limits_{j}
        \frac{(g_t^j - m_{t-1}^j)^2}{v_{t-1} + \epsilon} \Big)^{-1} \\
        m_t \leftarrow \frac{W_{t-1}}{W_{t-1} + w_t} m_{t-1} + \frac{w_t}{W_{t-1} + w_t} g_t \\
        v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) (g_t - g_{t-1})

    where :math:`g_t` is the gradient of :math:`\theta_t`,
    :math:`\beta_1, \beta_2 \in [0, 1]^2` are the exponential average smoothing coefficients,
    :math:`m_0 = 0,\ v_0 = 0,\ W_0 = \frac{\beta_1}{1 - \beta_1}`;
    :math:`\nu` is the degrees of freedom and :math:`d` if the number of dimensions of the parameter gradient.

    Then we correct their biases using:

    .. math::
        \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
        \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t}

    And finally the update step is performed using the following rule:

    .. math::
        \theta_t \leftarrow \theta_{t-1} - \alpha \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}

    where :math:`\theta_t` is the parameter value at step :math:`t` (:math:`\theta_0` being the initialization value),
    :math:`\alpha` is the learning rate, :math:`\epsilon > 0`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): coefficients used for running averages (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dof (int, optional): degrees of freedom
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        dof: Optional[float] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": amsgrad, "dof": dof}
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, torch.Tensor]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

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
            exp_avgs = []
            exp_avg_sqs = []
            W_ts = []  # noqa: N806
            max_exp_avg_sqs = []
            state_steps = []

            beta1, beta2 = group["betas"]

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
                        # Tadam specific
                        state["W_t"] = beta1 / (1 - beta1) * torch.ones(1, dtype=p.data.dtype, device=p.data.device)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    W_ts.append(state["W_t"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            tadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                W_ts,
                state_steps,
                group["amsgrad"],
                beta1,
                beta2,
                group["lr"],
                group["weight_decay"],
                group["eps"],
                group["dof"],
            )

        return loss


def tadam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    W_ts: List[Tensor],  # noqa: N803
    state_steps: List[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    dof: float,
) -> None:
    r"""Functional API that performs TAdam algorithm computation.
    See :class:`~holocron.optim.TAdam` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        W_t = W_ts[i]  # noqa: N806
        _dof = param.data.numel() if dof is None else dof
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        w_t = grad.sub(exp_avg).pow_(2).div_(exp_avg_sq.add(eps)).sum()
        w_t.add_(_dof).pow_(-1).mul_(_dof + param.data.numel())
        exp_avg.mul_(W_t / (W_t + w_t)).addcdiv_(w_t * grad, W_t + w_t)
        W_t.mul_((2 * beta1 - 1) / beta1)
        W_t.add_(w_t)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
