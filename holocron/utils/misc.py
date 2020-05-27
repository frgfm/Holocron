# -*- coding: utf-8 -*-

"""
Misc utilities
"""

from tqdm.auto import tqdm
import torch

__all__ = ['lr_finder']


def lr_finder(batch_training_fn, model, train_loader, optimizer, criterion, device=None,
              start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, stop_threshold=2, beta=0.9):
    """Learning rate finder as described in
    `"Cyclical Learning Rates for Training Neural Networks" <https://arxiv.org/pdf/1506.01186.pdf>`_

    Args:
        batch_training_fn (float): function used to train a model for a step
        model (torch.Tensor): model to train
        train_loader (torch.utils.data.DataLoader): training dataloader
        optimizer (torch.optim.Optimizer): model parameter optimizer
        criterion (nn.Module): loss computation function
        device (torch.device, optional): device to perform iterations on
        start_lr (float): initial learning rate
        end_lr (float): peak learning rate
        num_it (int): number of iterations to perform
        stop_div (bool): should the evaluation be stopped if loss diverges
        stop_threshold (float): if stop_div is True, stops the evaluation when loss reaches stop_threshold * best_loss
        beta (float): smoothing parameter for loss
    Returns:
        lrs (list<float>): list of used learning rates
        losses (list<float>): list of training losses
    """

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    model.train()
    model = model.to(device)
    loader_iter = iter(train_loader)
    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    base_lr = start_lr
    avg_loss, best_loss = 0, 0.
    lrs, losses = [], []
    for batch_idx in tqdm(range(num_it)):

        x, target = loader_iter.next()
        # Train for an iteration
        batch_loss = batch_training_fn(model, x, target, optimizer, criterion, device)

        if not isinstance(batch_loss, float):
            batch_loss = batch_loss.item()

        # Record loss
        avg_loss = beta * avg_loss + (1 - beta) * batch_loss
        smoothed_loss = avg_loss / (1 - beta ** (batch_idx + 1))
        if batch_idx == 0 or smoothed_loss < best_loss:
            best_loss = smoothed_loss

        if stop_div and batch_idx > 0 and (smoothed_loss > stop_threshold * best_loss):
            break
        lrs.append(base_lr)
        losses.append(smoothed_loss)

        # Update LR
        base_lr *= gamma
        for group in optimizer.param_groups:
            group['lr'] *= gamma

    return lrs, losses
