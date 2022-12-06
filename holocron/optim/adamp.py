# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Adam

from . import functional as F


class AdamP(Adam):
    r"""Implements the AdamP optimizer from `"AdamP: Slowing Down the Slowdown for Momentum Optimizers on
    Scale-invariant Weights" <https://arxiv.org/pdf/2006.08217.pdf>`_.

    The estimation of momentums is described as follows, :math:`\forall t \geq 1`:

    .. math::
        m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2

    where :math:`g_t` is the gradient of :math:`\theta_t`,
    :math:`\beta_1, \beta_2 \in [0, 1]^3` are the exponential average smoothing coefficients,
    :math:`m_0 = g_0,\ v_0 = 0`.

    Then we correct their biases using:

    .. math::
        \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t} \\
        \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t}

    And finally the update step is performed using the following rule:

    .. math::
        p_t \leftarrow \frac{\hat{m_t}}{\sqrt{\hat{n_t} + \epsilon}} \\
        q_t \leftarrow \begin{cases}
          \prod_{\theta_t}(p_t) & if\ cos(\theta_t, g_t) < \delta / \sqrt{dim(\theta)}\\
          p_t & \text{otherwise}\\
        \end{cases} \\
        \theta_t \leftarrow \theta_{t-1} - \alpha q_t

    where :math:`\theta_t` is the parameter value at step :math:`t` (:math:`\theta_0` being the initialization value),
    :math:`\prod_{\theta_t}(p_t)` is the projection of :math:`p_t` onto the tangent space of :math:`\theta_t`,
    :math:`cos(\theta_t, g_t)` is the cosine similarity between :math:`\theta_t` and :math:`g_t`,
    :math:`\alpha` is the learning rate, :math:`\delta > 0`, :math:`\epsilon > 0`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): coefficients used for running averages (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
        delta (float, optional): delta threshold for projection (default: False)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        delta: float = 0.1,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.delta = delta

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
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
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            beta1, beta2 = group["betas"]
            F.adamp(
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
            )

        return loss
