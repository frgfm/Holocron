#!usr/bin/python
# -*- coding: utf-8 -*-

'''
Lookahead optimizer wrapper
Inspired by https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
'''

import itertools as it
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    """Implements the Lookahead optimizer wrapper from https://arxiv.org/pdf/1907.08610.pdf

    Args:
        base_optimizer (torch.optim.optimizer.Optimizer): base parameter optimizer
        alpha (int, optional): rate of weight synchronization
        k (int, optional): number of step performed on fast weights before weight synchronization
    """

    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if alpha < 0 or alpha > 1:
            raise ValueError(f'expected positive float lower than 1 as synchronization rate alpha, received: {alpha}')
        if not isinstance(k, int) or k < 1:
            raise ValueError(f'expected positive integer as synchronization period k, received: {k}')
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        # Slow parameters
        self.slow_groups = [[p.clone().detach() for p in group['params']]
                             for group in self.param_groups]
        for w in it.chain(*self.slow_groups):
            w.requires_grad = False

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update fast params
        loss = self.base_optimizer.step()
        self.step_counter += 1
        # Synchronization every k steps on fast params
        if self.step_counter % self.k == 0:
            for fast_group, slow_group in zip(self.param_groups, self.slow_groups):
                for fast_p, slow_p in zip(fast_group['params'], slow_group):
                    # Outer update
                    slow_p.data.add_(self.alpha, fast_p.data - slow_p.data)
                    # Synchronize fast and slow params
                    fast_p.data.copy_(slow_p.data)

        return loss

    def __getstate__(self):
        return self.base_optimizer.__getstate__()

    def __setstate__(self, state):
        self.base_optimizer.__setstate__(state)

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        optimizer_repr = self.base_optimizer.__repr__().replace('\n', '\n\t')
        format_string += f"\nbase_optimizer={optimizer_repr},"
        format_string += f"\nalpha={self.alpha},"
        format_string += f"\nk={self.k}"
        format_string += '\n)'
        return format_string
