# -*- coding: utf-8 -*-

'''
Extended version of Adam optimizer with Student-t mean estimation
'''

import torch
from torch.optim.optimizer import Optimizer


class TAdam(Optimizer):
    """Implements the TAdam optimizer from `"TAdam: A Robust Stochastic Gradient Optimizer"
    <https://arxiv.org/pdf/2003.00179.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): coefficients used for running averages (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dof (int, optional): degrees of freedom
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, dof=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, dof=dof)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
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
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    #
                    state['W_t'] = beta1 / (1 - beta1)
                    state['d'] = p.data.numel()
                    state['dof'] = state['d'] if group['dof'] is None else group['dof']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                wt = grad.sub(exp_avg).pow_(2).div_(exp_avg_sq.add(group['eps'])).sum()
                wt.add_(state['dof']).pow_(-1).mul_(state['dof'] + state['d'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(state['W_t'] / (state['W_t'] + wt)).add_(grad, alpha=wt / (state['W_t'] + wt))
                state['W_t'] *= (2 * beta1 - 1) / beta1
                state['W_t'] += wt
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

                # Adaptive momentum
                p.data.addcdiv_(exp_avg / bias_correction1,
                                (exp_avg_sq / bias_correction2).sqrt().add_(group['eps']), value=-group['lr'])

        return loss
