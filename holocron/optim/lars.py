# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["LARS"]


class LARS(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.

    The estimation of global and local learning rates is described as follows, :math:`\forall t \geq 1`:

    .. math::
        \alpha_t \leftarrow \alpha (1 - t / T)^2 \\
        \gamma_t \leftarrow \frac{\lVert \theta_t \rVert}{\lVert g_t \rVert  + \lambda \lVert \theta_t \rVert}

    where :math:`\theta_t` is the parameter value at step :math:`t` (:math:`\theta_0` being the initialization value),
    :math:`g_t` is the gradient of :math:`\theta_t`,
    :math:`T` is the total number of steps,
    :math:`\alpha` is the learning rate
    :math:`\lambda \geq 0` is the weight decay.

    Then we estimate the momentum using:

    .. math::
        v_t \leftarrow m v_{t-1} + \alpha_t \gamma_t (g_t + \lambda \theta_t)

    where :math:`m` is the momentum and :math:`v_0 = 0`.

    And finally the update step is performed using the following rule:

    .. math::
        \theta_t \leftarrow \theta_{t-1} - v_t

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        scale_clip (tuple, optional): the lower and upper bounds for the weight norm in local LR of LARS
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        scale_clip: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        # LARS arguments
        self.scale_clip = scale_clip
        if self.scale_clip is None:
            self.scale_clip = (0.0, 10.0)

    def __setstate__(self, state: Dict[str, torch.Tensor]):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

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
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # LARS
                p_norm = torch.norm(p.data)
                denom = torch.norm(d_p)
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    denom.add_(p_norm, alpha=weight_decay)
                # Compute the local LR
                if p_norm == 0 or denom == 0:
                    local_lr = 1
                else:
                    local_lr = p_norm / denom

                if momentum == 0:
                    p.data.add_(d_p, alpha=-group["lr"] * local_lr)
                else:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        momentum_buffer = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        momentum_buffer = param_state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum_buffer, alpha=momentum)
                    else:
                        d_p = momentum_buffer
                    p.data.add_(d_p, alpha=-group["lr"] * local_lr)
                    self.state[p]["momentum_buffer"] = momentum_buffer

        return loss
