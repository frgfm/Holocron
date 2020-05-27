# -*- coding: utf-8 -*-

'''
Lookahead optimizer wrapper
'''

import torch
from collections import defaultdict
from torch.optim.optimizer import Optimizer


__all__ = ['Lookahead', 'Scout']


class Lookahead(Optimizer):
    """Implements the Lookahead optimizer wrapper from `"Lookahead Optimizer: k steps forward, 1 step back"
    <https://arxiv.org/pdf/1907.08610.pdf>`_.

    Args:
        base_optimizer (torch.optim.optimizer.Optimizer): base parameter optimizer
        sync_rate (int, optional): rate of weight synchronization
        sync_period (int, optional): number of step performed on fast weights before weight synchronization
    """

    def __init__(self, base_optimizer, sync_rate=0.5, sync_period=6):
        if sync_rate < 0 or sync_rate > 1:
            raise ValueError(f'expected positive float lower than 1 as sync_rate, received: {sync_rate}')
        if not isinstance(sync_period, int) or sync_period < 1:
            raise ValueError(f'expected positive integer as sync_period, received: {sync_period}')
        # Optimizer attributes
        self.defaults = dict(sync_rate=sync_rate, sync_period=sync_period)
        self.state = defaultdict(dict)
        # Base optimizer attributes
        self.base_optimizer = base_optimizer
        # Wrapper attributes
        self.fast_steps = 0
        self.param_groups = []
        for group in self.base_optimizer.param_groups:
            self._add_param_group(group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'base_state': self.base_optimizer.__getstate__(),
            'fast_steps': self.fast_steps,
            'param_groups': self.param_groups
        }

    def state_dict(self):
        return dict(**super(Lookahead, self).state_dict(),
                    base_state_dict=self.base_optimizer.state_dict())

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict['base_state_dict'])
        super(Lookahead, self).load_state_dict(state_dict)
        # Update last key of class dict
        self.__setstate__({'base_state_dict': self.base_optimizer.state_dict()})

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        # Update fast params
        loss = self.base_optimizer.step(closure)
        self.fast_steps += 1
        # Synchronization every sync_period steps on fast params
        if self.fast_steps % self.defaults['sync_period'] == 0:
            self.sync_params(self.defaults['sync_rate'])

        return loss

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        optimizer_repr = self.base_optimizer.__repr__().replace('\n', '\n\t')
        format_string += f"\nbase_optimizer={optimizer_repr},"
        for arg, val in self.defaults.items():
            format_string += f"\n{arg}={val},"
        format_string += '\n)'
        return format_string

    def _add_param_group(self, param_group):
        """Adds a new slow parameter group

        Args:
            param_group (dict): parameter group of base_optimizer
        """

        # Clone & detach params from base optimizer
        group = dict(params=[p.clone().detach()
                             for p in param_group['params']],
                     lr=param_group['lr'])
        # Uneeded grads
        for p in group['params']:
            p.reguires_grad = False
        self.param_groups.append(group)

    def add_param_group(self, param_group):
        """Adds a parameter group to base optimizer (fast weights) and its corresponding slow version

        Args:
            param_group (dict): parameter group
        """

        # Add param group to base optimizer
        self.base_optimizer.add_param_group(param_group)

        # Add the corresponding slow param group
        self._add_param_group(self.base_optimizer.param_groups[-1])

    def sync_params(self, sync_rate=0):
        """Synchronize parameters as follows:
        slow_param <- slow_param + sync_rate * (fast_param - slow_param)

        Args:
            sync_rate (float): synchronization rate of parameters
        """

        for fast_group, slow_group in zip(self.base_optimizer.param_groups, self.param_groups):
            for fast_p, slow_p in zip(fast_group['params'], slow_group['params']):
                # Outer update
                if sync_rate > 0:
                    slow_p.data.add_(fast_p.data - slow_p.data, alpha=sync_rate)
                # Synchronize fast and slow params
                fast_p.data.copy_(slow_p.data)


class Scout(Optimizer):
    """Implements a new optimizer wrapper based on `"Lookahead Optimizer: k steps forward, 1 step back"
    <https://arxiv.org/pdf/1907.08610.pdf>`_.

    Args:
        base_optimizer (torch.optim.optimizer.Optimizer): base parameter optimizer
        sync_rate (int, optional): rate of weight synchronization
        sync_period (int, optional): number of step performed on fast weights before weight synchronization
    """

    def __init__(self, base_optimizer, sync_rate=0.5, sync_period=6):
        if sync_rate < 0 or sync_rate > 1:
            raise ValueError(f'expected positive float lower than 1 as sync_rate, received: {sync_rate}')
        if not isinstance(sync_period, int) or sync_period < 1:
            raise ValueError(f'expected positive integer as sync_period, received: {sync_period}')
        # Optimizer attributes
        self.defaults = dict(sync_rate=sync_rate, sync_period=sync_period)
        self.state = defaultdict(dict)
        # Base optimizer attributes
        self.base_optimizer = base_optimizer
        # Wrapper attributes
        self.fast_steps = 0
        self.param_groups = []
        for group in self.base_optimizer.param_groups:
            self._add_param_group(group)
        # Buffer for scouting
        self.buffer = []
        for group in self.param_groups:
            for p in group['params']:
                self.buffer.append(p.data.unsqueeze(0))

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'base_state': self.base_optimizer.__getstate__(),
            'fast_steps': self.fast_steps,
            'param_groups': self.param_groups
        }

    def state_dict(self):
        return dict(**super(Scout, self).state_dict(),
                    base_state_dict=self.base_optimizer.state_dict())

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict['base_state_dict'])
        super(Scout, self).load_state_dict(state_dict)
        # Update last key of class dict
        self.__setstate__({'base_state_dict': self.base_optimizer.state_dict()})

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        # Update fast params
        loss = self.base_optimizer.step(closure)
        self.fast_steps += 1
        # Add it to buffer
        idx = 0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.buffer[idx] = torch.cat((self.buffer[idx], p.data.clone().detach().unsqueeze(0)))
                idx += 1
        # Synchronization every sync_period steps on fast params
        if self.fast_steps % self.defaults['sync_period'] == 0:
            # Compute STD of updates
            update_similarity = []
            for _ in range(len(self.buffer)):
                p = self.buffer.pop()
                update = p[1:] - p[:-1]
                max_dev = (update - torch.mean(update, dim=0)).abs().max(dim=0).values
                update_similarity.append((torch.std(update, dim=0) / max_dev).mean().item())
            update_coherence = sum(std_list) / len(std_list)

            sync_rate = max(1 - update_coherence, self.defaults['sync_rate'])
            # sync_rate = self.defaults['sync_rate']
            self.sync_params(sync_rate)
            # Reset buffer
            self.buffer = []
            for group in self.param_groups:
                for p in group['params']:
                    self.buffer.append(p.data.unsqueeze(0))

        return loss

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        optimizer_repr = self.base_optimizer.__repr__().replace('\n', '\n\t')
        format_string += f"\nbase_optimizer={optimizer_repr},"
        for arg, val in self.defaults.items():
            format_string += f"\n{arg}={val},"
        format_string += '\n)'
        return format_string

    def _add_param_group(self, param_group):
        """Adds a new slow parameter group

        Args:
            param_group (dict): parameter group of base_optimizer
        """

        # Clone & detach params from base optimizer
        group = dict(params=[p.clone().detach()
                             for p in param_group['params']],
                     lr=param_group['lr'])
        # Uneeded grads
        for p in group['params']:
            p.reguires_grad = False
        self.param_groups.append(group)

    def add_param_group(self, param_group):
        """Adds a parameter group to base optimizer (fast weights) and its corresponding slow version

        Args:
            param_group (dict): parameter group
        """

        # Add param group to base optimizer
        self.base_optimizer.add_param_group(param_group)

        # Add the corresponding slow param group
        self._add_param_group(self.base_optimizer.param_groups[-1])

    def sync_params(self, sync_rate=0):
        """Synchronize parameters as follows:
        slow_param <- slow_param + sync_rate * (fast_param - slow_param)

        Args:
            sync_rate (float): synchronization rate of parameters
        """

        for fast_group, slow_group in zip(self.base_optimizer.param_groups, self.param_groups):
            for fast_p, slow_p in zip(fast_group['params'], slow_group['params']):
                # Outer update
                if sync_rate > 0:
                    slow_p.data.add_(fast_p.data - slow_p.data, alpha=sync_rate)
                # Synchronize fast and slow params
                fast_p.data.copy_(slow_p.data)
