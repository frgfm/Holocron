import math
import torch
from torch import Tensor
from typing import List


def radam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float
) -> None:
    r"""Functional API that performs RAdam algorithm computation.
    See :class:`~holocron.optim.RAdam` for details.
    """

    sma_inf = 2 / (1 - beta2) - 1

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute length of SMA
        sma_t = sma_inf - 2 * step * (1 - bias_correction2) / bias_correction2
        step_size = lr / bias_correction1

        if sma_t > 4:
            # Variance rectification term
            r_t = math.sqrt((sma_t - 4) * (sma_t - 2) * sma_inf / ((sma_inf - 4) * (sma_inf - 2) * sma_t))
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            # Adaptive momentum
            param.data.addcdiv_(exp_avg, denom, value=-r_t * step_size)
        else:
            # Unadapted momentum
            param.data.add_(exp_avg, alpha=-step_size)


def tadam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    W_ts: List[Tensor],
    state_steps: List[int],
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    dof: float
) -> None:
    r"""Functional API that performs TAdam algorithm computation.
    See :class:`~holocron.optim.TAdam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        W_t = W_ts[i]
        _dof = param.data.numel() if dof is None else dof
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

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
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)  # type: ignore[attr-defined]
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


def adabelief(
    params: List[Tensor],
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
    eps: float
) -> None:
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
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)  # type: ignore[attr-defined]
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
