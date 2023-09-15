# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Dict

import torch

from .core import Trainer

__all__ = ["SegmentationTrainer"]


class SegmentationTrainer(Trainer):
    """Semantic segmentation trainer class.

    Args:
        model: model to train
        train_loader: training loader
        val_loader: validation loader
        criterion: loss criterion
        optimizer: parameter optimizer
        gpu: index of the GPU to use
        output_file: path where checkpoints will be saved
        amp: whether to use automatic mixed precision
        skip_nan_loss: whether the optimizer step should be skipped when the loss is NaN
        nan_tolerance: number of consecutive batches with NaN loss before stopping the training
        gradient_acc: number of batches to accumulate the gradient of before performing the update step
        gradient_clip: the gradient clip value
        on_epoch_end: callback triggered at the end of an epoch
    """

    def __init__(self, *args: Any, num_classes: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    @torch.inference_mode()
    def evaluate(self, ignore_index: int = 255) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Args:
            ignore_index (int, optional): index of the class to ignore in evaluation

        Returns:
            dict: evaluation metrics
        """
        self.model.eval()

        val_loss, mean_iou, num_valid_batches = 0.0, 0.0, 0
        conf_mat = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=next(self.model.parameters()).device
        )
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            _loss, out = self._get_loss(x, target, return_logits=True)

            # Safeguard for NaN loss
            if not torch.isnan(_loss) and not torch.isinf(_loss):
                val_loss += _loss.item()
                num_valid_batches += 1

            # borrowed from https://github.com/pytorch/vision/blob/master/references/segmentation/train.py
            pred = out.argmax(dim=1).flatten()
            target = target.flatten()
            k = (target >= 0) & (target < self.num_classes)
            inds = self.num_classes * target[k].to(torch.int64) + pred[k]
            nc = self.num_classes
            conf_mat += torch.bincount(inds, minlength=nc**2).reshape(nc, nc)

        val_loss /= num_valid_batches
        acc_global = (torch.diag(conf_mat).sum() / conf_mat.sum()).item()
        mean_iou = (torch.diag(conf_mat) / (conf_mat.sum(1) + conf_mat.sum(0) - torch.diag(conf_mat))).mean().item()

        return {"val_loss": val_loss, "acc_global": acc_global, "mean_iou": mean_iou}

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return (
            f"Validation loss: {eval_metrics['val_loss']:.4} "
            f"(Acc: {eval_metrics['acc_global']:.2%} | Mean IoU: {eval_metrics['mean_iou']:.2%})"
        )
