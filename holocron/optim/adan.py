# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Adam

from . import functional as F


class Adan(Adam):
    """Implements the Adan optimizer from `"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep
    Models" <https://arxiv.org/pdf/2208.06677.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float, float], optional): coefficients used for running averages
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
        delta (float, optional): delta threshold for projection (default: False)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],  # type: ignore[name-defined]
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

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
            F.adan(
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
