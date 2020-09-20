import math
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, MultiplicativeLR
from fastprogress import master_bar, progress_bar

from contiguous_params import ContiguousParams


__all__ = ['Trainer', 'ClassificationTrainer']


class Trainer:

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 gpu_id=0, output_file=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        # Output folder
        self.output_file = output_file

        # Initialize
        self.step = 0
        self.start_epoch = 0
        self.epoch = 0
        self.min_loss = math.inf
        self.set_device(gpu_id)

    def set_device(self, gpu_id):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        torch.cuda.set_device(gpu_id)
        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()

    def load(self, state):
        """Resume from a trainer state"""
        self.start_epoch = state['epoch']
        self.epoch = self.start_epoch
        self.step = state['step']
        self.min_loss = state['min_loss']
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.model.load_state_dict(state['model'])

    def _fit_epoch(self, freeze_until, mb):
        """Fit a single epoch"""
        self.model = freeze_bn(self.model.train())

        pb = progress_bar(self.train_loader, parent=mb)
        for x, target in pb:
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss = self._get_loss(x, target)

            # Backprop
            self._backprop_step(batch_loss)
            # Update LR
            self.scheduler.step()
            pb.comment = f"Training loss: {batch_loss.item():.4}"

            self.step += 1
        self.epoch += 1

    @staticmethod
    def to_cuda(x, target):
        """Move input and target to GPU"""
        raise NotImplementedError

    def _backprop_step(self, loss):
        """Compute error gradients & perform the optimizer step"""
        raise NotImplementedError

    def _get_loss(self, x, target):
        """Forward tensor and compute the loss"""
        raise NotImplementedError

    def _reset_opt(self, params, lr):
        """Reset the target params of the optimizer"""
        self.optimizer.defaults['lr'] = lr
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups = []
        self.optimizer.add_param_group(dict(params=params))

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    def _eval_metrics_str(eval_metrics):
        return ""

    def _reset_scheduler(self, lr, num_epochs, sched_type='onecycle'):
        if sched_type == 'onecycle';
            self.scheduler = OneCycleLR(self.optimizer, lr, num_epochs * len(self.train_loader))
        elif sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, num_epochs * len(self.train_loader), eta_min=lr / 25e4)
        else:
            raise ValueError(f"The following scheduler type is not supported: {scheduler_type}")

    def fit_n_epochs(self, num_epochs, lr, freeze_until, sched_type='onecycle'):

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        params = ContiguousParams([p for p in self.model.parameters() if p.requires_grad])
        self._reset_opt(params.contiguous(), lr)
        # Scheduler
        self._reset_scheduler(lr, num_epochs, sched_type)

        mb = master_bar(range(num_epochs))
        for epoch_idx in mb:

            self._fit_epoch(freeze_until, mb)
            # Check whether ops invalidated the buffer
            params.assert_buffer_is_valid()
            eval_metrics = self.evaluate()

            # master bar
            mb.main_bar.comment = f"Epoch {self.start_epoch + self.epoch}/{self.start_epoch+num_epochs}"
            mb.write(f"Epoch {self.start_epoch + self.epoch}/{self.start_epoch+num_epochs} - "
                     f"{self._eval_metrics_str(eval_metrics)}")

            if eval_metrics['val_loss'] < self.min_loss:
                print(f"Validation loss decreased {self.min_loss:.4} --> "
                      f"{eval_metrics['val_loss']:.4}: saving state...")
                self.min_loss = eval_metrics['val_loss']
                self.save(self.output_file)

    def save(self, output_file):
        torch.save(dict(epoch=self.epoch, step=self.step, min_loss=self.min_loss,
                        optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                        model=self.model.state_dict()),
                   output_file,
                   _use_new_zipfile_serialization=False)

    def lr_find(self, freeze_until, start_lr=1e-7, end_lr=1, num_it=100):

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        params = ContiguousParams([p for p in self.model.parameters() if p.requires_grad])
        self._reset_opt(params.contiguous(), start_lr)
        gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
        scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

        self.lr_recorder = [start_lr * gamma ** idx for idx in range(num_it)]
        self.loss_recorder = []

        for batch_idx, (x, target) in enumerate(self.train_loader):
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss = self._get_loss(x, target)
            self._backprop_step(batch_loss)
            # Update LR
            scheduler.step()

            # Record
            self.loss_recorder.append(batch_loss.item())
            # Stop after the number of iterations
            if batch_idx + 1 == num_it:
                break

    def plot_recorder(self, stop_div=True, stop_threshold=3, beta=0.95):

        if len(self.lr_recorder) != len(self.loss_recorder) or len(self.lr_recorder) == 0:
            raise AssertionError("Please run the `lr_find` method first")

        # Exp moving average of loss
        smoothed_losses = []
        avg_loss = 0
        for idx, loss in enumerate(self.loss_recorder):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

        plt.plot(self.lr_recorder[10:-5], smoothed_losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Training loss')
        plt.grid(True, linestyle='--', axis='x')
        plt.show(block=True)

    def show_worst_preds(self, n=4):
        # Loop on training data, order decreasing loss
        return NotImplementedError

    def half(self):
        """Switch to FP16"""
        return NotImplementedError

    def check_setup(self, lr=3e-4, num_it=100):
        """Check whether you can overfit one batch"""

        self.model = freeze_bn(self.model.train())
        # Update param groups & LR
        params = ContiguousParams([p for p in self.model.parameters() if p.requires_grad])
        self._reset_opt(params.contiguous(), lr)

        prev_loss = math.inf

        x, target = self.to_cuda(next(iter(self.train_loader)))

        for _ in range(num_it):
            # Forward
            batch_loss = self._get_loss(x, target)
            # Backprop
            self._backprop_step(batch_loss)

            # Check that loss decreases
            if batch_loss.item() > prev_loss:
                raise AssertionError("Unable to overfit one batch. Please investigate")
            prev_loss = batch_loss.item()

        return True


class ClassificationTrainer(Trainer):

    @staticmethod
    def to_cuda(x, target):
        """Move input and target to GPU"""
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        return x, target

    def _backprop_step(self, loss):
        # Clean gradients
        self.optimizer.zero_grad()
        # Backpropate the loss
        loss.backward()
        # Update the params
        self.optimizer.step()

    def _get_loss(self, x, target):
        # Forward
        out = self.model(x)
        # Loss computation
        return self.criterion(out, target)

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        val_loss, top1, top5, num_samples = 0, 0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            val_loss += self._get_loss(x, target).item()

            pred = out.topk(5, dim=1)[1]
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            top1 += correct[:, 0].sum().item()
            top5 += correct.any(dim=1).sum().item()
            num_samples += x.shape[0]

        val_loss /= len(self.val_loader)

        return dict(val_loss=val_loss, acc1=top1 / num_samples, acc5=top5 / num_samples)

    @staticmethod
    def _eval_metrics_str(eval_metrics):
        return (f"Validation loss: {eval_metrics['val_loss']:.4} "
                f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})")
