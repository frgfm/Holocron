# -*- coding: utf-8 -*-


"""
Optimization schedulers
"""

import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


__all__ = ['OneCycleScheduler']


class OneCycleScheduler(_LRScheduler):
    """Implements the One Cycle scheduler from `"A disciplined approach to neural network hyper-parameters"
    <https://arxiv.org/pdf/1803.09820.pdf>`_. Please note that this implementation was made before pytorch supports it,
    using the official Pytorch implementation is advised.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_size (int): Number of training iterations to be performed
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        warmup_ratio (float): ratio of iterations used to reach max_lr
        phases (tuple): specify the scaling mode of both phases (possible values: 'linear', 'cosine')
        base_ratio (float): ratio between base_lr and max_lr during warmup phase
        final_ratio (float): ratio between base_lr and max_lr during last phase
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
    """

    def __init__(self,
                 optimizer,
                 total_size,
                 max_lr=None,
                 warmup_ratio=0.3,
                 phases=None,
                 base_ratio=0.2,
                 final_ratio=None,
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Specify max lr
        if isinstance(max_lr, float):
            self.max_lrs = [max_lr for group in optimizer.param_groups]
        elif isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} values for max_lr, got {len(max_lr)}")
            self.max_lrs = max_lr
        else:
            # Take current value as max_lr
            self.max_lrs = [group['lr'] for group in optimizer.param_groups]

        # Take the division factor for each phase
        self.base_ratio = base_ratio
        self.final_ratio = base_ratio * 1e-4 if final_ratio is None else final_ratio

        self.total_size = total_size
        self.warmup_ratio = warmup_ratio

        # Phases
        self.phases = phases if isinstance(phases, tuple) else ('linear', 'cosine')
        modes = ['linear', 'cosine']
        if any(phase not in modes for phase in self.phases):
            raise ValueError(f"Phases can only take values from {modes}")

        # Handle momentum for specific optimizer
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
            self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        step_ratio = self.last_epoch / self.total_size
        # Get phase progress and LR divider for current phase
        if step_ratio <= self.warmup_ratio:
            phase_idx = 0
            x = step_ratio / self.warmup_ratio
            base_ratio = self.base_ratio
        else:
            phase_idx = 1
            x = (step_ratio - self.warmup_ratio) / (1 - self.warmup_ratio)
            base_ratio = self.final_ratio

        # Adapt scaling based on phase mode
        if self.phases[phase_idx] == 'linear':
            scale_factor = x
        elif self.phases[phase_idx] == 'cosine':
            scale_factor = 0.5 * (1 + math.cos(x * math.pi))

        # Populate LR for each group
        lrs = []
        for max_lr in self.max_lrs:
            base_lr = base_ratio * max_lr
            base_height = (max_lr - base_lr) * scale_factor
            lrs.append(base_lr + base_height)

        # Populate momentum for each group
        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_momentum = base_ratio * max_momentum
                base_height = (max_momentum - base_momentum) * scale_factor
                momentums.append(max_momentum - base_height)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

    def __repr__(self):
        return (f"{self.__class__.__name__}(max_lr={max(self.max_lrs)}, warmup_ratio={self.warmup_ratio}, "
                f"base_ratio={self.base_ratio}, final_ratio={self.final_ratio}, phases={self.phases})")
