# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class RaLars(Optimizer):
    """Implements the RAdam optimizer from `"On the variance of the Adaptive Learning Rate and Beyond"
    <https://arxiv.org/pdf/1908.03265.pdf>`_ with optional Layer-wise adaptive Scaling from
    `"Large Batch Training of Convolutional Networks" <https://arxiv.org/pdf/1708.03888.pdf>`_

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): coefficients used for running averages  (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        force_adaptive_momentum (float, optional): use adaptive momentum if variance is not tractable (default: False)
        scale_clip (float, optional): the maximal upper bound for the scale factor of LARS
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],  # type: ignore[name-defined]
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        force_adaptive_momentum: bool = False,
        scale_clip: Optional[Tuple[float, float]] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RaLars, self).__init__(params, defaults)
        # RAdam tweaks
        self.force_adaptive_momentum = force_adaptive_momentum
        # LARS arguments
        self.scale_clip = scale_clip
        if self.scale_clip is None:
            self.scale_clip = (0, 10)

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

            # Get group-shared variables
            beta1, beta2 = group["betas"]
            # Compute max length of SMA on first step
            if not isinstance(group.get("sma_inf"), float):
                group["sma_inf"] = 2 / (1 - beta2) - 1
            sma_inf = group["sma_inf"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(f"{self.__class__.__name__} does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute length of SMA
                sma_t = sma_inf - 2 * state["step"] * (1 - bias_correction2) / bias_correction2

                update = torch.zeros_like(p.data)
                if sma_t > 4:
                    # Variance rectification term
                    r_t = math.sqrt((sma_t - 4) * (sma_t - 2) * sma_inf / ((sma_inf - 4) * (sma_inf - 2) * sma_t))
                    # Adaptive momentum
                    update.addcdiv_(
                        exp_avg / bias_correction1, (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"]), value=r_t
                    )
                else:
                    if self.force_adaptive_momentum:
                        # Adaptive momentum without variance rectification (Adam)
                        update.addcdiv_(
                            exp_avg / bias_correction1, (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                        )
                    else:
                        # Unadapted momentum
                        update.add_(exp_avg / bias_correction1)

                # Weight decay
                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                # LARS
                p_norm = p.data.pow(2).sum().sqrt()
                update_norm = update.pow(2).sum().sqrt()
                phi_p = p_norm.clamp(*self.scale_clip)
                # Compute the local LR
                if phi_p == 0 or update_norm == 0:
                    local_lr = 1
                else:
                    local_lr = phi_p / update_norm

                state["local_lr"] = local_lr

                p.data.add_(update, alpha=-group["lr"] * local_lr)

        return loss
