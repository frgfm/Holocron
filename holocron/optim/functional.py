r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List


def adabelief(params: List[Tensor],
              grads: List[Tensor],
              exp_avgs: List[Tensor],
              exp_avg_sqs: List[Tensor],
              max_exp_avg_sqs: List[Tensor],
              state_steps: List[int],
              amsgrad: bool,
              beta1: float,
              beta2: float,
              lr: float,
              weight_decay: float,
              eps: float):
    r"""Functional API that performs AdaBelief algorithm computation.
    See :class:`~holocron.optim.AdaBelief` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        grad_residual = grad - exp_avg
        exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
