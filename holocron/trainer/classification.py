# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict

import torch

from .core import Trainer

__all__ = ['ClassificationTrainer', 'BinaryClassificationTrainer']


class ClassificationTrainer(Trainer):
    """Image classification trainer class

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (torch.nn.Module): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
        amp (bool, optional): whether to use automatic mixed precision
    """

    @torch.inference_mode()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Returns:
            dict: evaluation metrics
        """

        self.model.eval()

        val_loss, top1, top5, num_samples, num_valid_batches = 0., 0, 0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            if self.amp:
                with torch.cuda.amp.autocast():
                    # Forward
                    out = self.model(x)
                    # Loss computation
                    _loss = self.criterion(out, target)
            else:
                # Forward
                out = self.model(x)
                # Loss computation
                _loss = self.criterion(out, target)

            # Safeguard for NaN loss
            if not torch.isnan(_loss) and not torch.isinf(_loss):
                val_loss += _loss.item()
                num_valid_batches += 1

            pred = out.topk(5, dim=1)[1] if out.shape[1] >= 5 else out.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            top1 += correct[:, 0].sum().item()
            if out.shape[1] >= 5:
                top5 += correct.any(dim=1).sum().item()

            num_samples += x.shape[0]

        val_loss /= num_valid_batches

        return dict(val_loss=val_loss, acc1=top1 / num_samples, acc5=top5 / num_samples)

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return (f"Validation loss: {eval_metrics['val_loss']:.4} "
                f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})")


class BinaryClassificationTrainer(Trainer):
    """Image binary classification trainer class

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (torch.nn.Module): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
        amp (bool, optional): whether to use automatic mixed precision
    """

    @torch.inference_mode()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Returns:
            dict: evaluation metrics
        """

        self.model.eval()

        val_loss, top1, num_samples, num_valid_batches = 0., 0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            if self.amp:
                with torch.cuda.amp.autocast():
                    # Forward
                    out = self.model(x)
                    # Apply sigmoid
                    out = torch.sigmoid(out)
                    # Loss computation
                    _loss = self.criterion(out, target)
            else:
                # Forward
                out = self.model(x)
                # Apply sigmoid
                out = torch.sigmoid(out)
                # Loss computation
                _loss = self.criterion(out, target)

            # Safeguard for NaN loss
            if not torch.isnan(_loss) and not torch.isinf(_loss):
                val_loss += _loss.item()
                num_valid_batches += 1

            top1 += int(torch.sum((target >= 0.5) == (out >= 0.5)).item())

            num_samples += x.shape[0]

        val_loss /= num_valid_batches

        return dict(val_loss=val_loss, acc=top1 / num_samples)

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return (f"Validation loss: {eval_metrics['val_loss']:.4} "
                f"(Acc: {eval_metrics['acc']:.2%})")
